import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading
import traceback


# Keep native libraries and Python diagnostics off the JSON protocol pipe.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

UNET = "qwen_image_2.1_int8_convrot.safetensors"
CLIP = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def emit(event):
    protocol.write(json.dumps(event, ensure_ascii=False) + "\n")
    protocol.flush()


def workflow(prompt, images, seed, prefix):
    graph = {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["model", 0], "device": "gpu", "dtype": "default"}},
        "encode": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "negative_prompt": "", "resolution": 0}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["cache", 0], "positive": ["encode", 0], "negative": ["encode", 1], "latent_image": ["encode", 2], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["decode", 0], "filename_prefix": prefix, "format": "png", "format.bit_depth": "8-bit", "format.input_color_space": "sRGB"}},
    }
    for number, image in enumerate(images, 1):
        name = f"image_{number}"
        graph[name] = {"class_type": "LoadImage", "inputs": {"image": image}}
        graph["encode"]["inputs"][f"images.{name}"] = [name, 0]
    return graph


def run():
    incoming = b""
    while b"\n" not in incoming:
        chunk = os.read(sys.stdin.fileno(), 65536)
        if not chunk:
            raise RuntimeError("Edit closed the worker input before sending a batch")
        incoming += chunk
    batch, control = incoming.split(b"\n", 1)
    request = json.loads(batch)
    runtime = request["runtime"]
    working = Path(request["working"])
    os.chdir(working)

    # All implicit runtime files stay in this batch's project-local directory.
    for variable, directory in {
        "TEMP": "temp",
        "TMP": "temp",
        "TMPDIR": "temp",
        "XDG_CACHE_HOME": "cache",
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_ASSETS_CACHE": "cache/huggingface/assets",
        "HF_XET_CACHE": "cache/huggingface/xet",
        "HF_MODULES_CACHE": "cache/huggingface/modules",
        "TORCH_HOME": "cache/torch",
        "TORCH_EXTENSIONS_DIR": "cache/torch/extensions",
        "PYTORCH_KERNEL_CACHE_PATH": "cache/torch/kernels",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/cuda",
    }.items():
        path = working / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    sys.path.insert(0, runtime["comfyui"])
    sys.argv = [sys.argv[0], "--base-directory", str(working), "--disable-pinned-memory"]

    import comfy.options
    comfy.options.enable_args_parsing()
    import cuda_malloc
    import comfy_aimdo.control
    if not comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True):
        raise RuntimeError("Could not initialize ComfyUI dynamic VRAM")
    import torch
    import folder_paths
    import comfy.model_management
    import comfy.memory_management
    import comfy.model_patcher
    from comfy.cli_args import args
    import nodes
    from comfy_api.latest import io
    from comfy_extras.nodes_images import SaveImageAdvanced
    from comfy_extras.nodes_qwen import QwenImage21Cache, TextEncodeQwenImage21

    if not comfy_aimdo.control.init_devices(
        (device.index, int(args.vram_headroom * 1024 ** 3))
        for device in comfy.model_management.get_all_torch_devices()
    ):
        raise RuntimeError("Could not initialize ComfyUI dynamic VRAM devices")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    for category, path in runtime["models"].items():
        folder_paths.folder_names_and_paths[category] = ([path], folder_paths.supported_pt_extensions)
    folder_paths.set_output_directory(runtime["output"])

    def stop():
        command = control
        while b"\n" not in command:
            chunk = os.read(sys.stdin.fileno(), 512)
            if not chunk:
                break
            command += chunk
        comfy.model_management.interrupt_current_processing(True)

    threading.Thread(target=stop, daemon=True).start()
    try:
        with torch.inference_mode():
            comfy.model_management.throw_exception_if_processing_interrupted()
            emit({"type": "progress", "stage": "preparing", "file": ""})
            input_directory = Path(folder_paths.get_input_directory())
            input_names = {"images": [], "references": []}
            for group, names in input_names.items():
                for number, source in enumerate(request[group], 1):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    name = f"{group}_{number:04d}{Path(source).suffix}"
                    shutil.copyfile(source, input_directory / name)
                    names.append(name)
            model = nodes.UNETLoader().load_unet(UNET, "default")[0]
            model = QwenImage21Cache.execute(model, "gpu", "default").result[0]
            comfy.model_management.throw_exception_if_processing_interrupted()
            clip = nodes.CLIPLoader().load_clip(CLIP, "qwen_image", "default")[0]
            comfy.model_management.throw_exception_if_processing_interrupted()
            vae = nodes.VAELoader().load_vae(VAE)[0]
            references = {
                f"image_{number}": nodes.LoadImage().load_image(path)[0]
                for number, path in enumerate(input_names["references"], 2)
            }
            for source, name in zip(request["images"], input_names["images"]):
                comfy.model_management.throw_exception_if_processing_interrupted()
                emit({"type": "progress", "stage": "editing", "file": source})
                images = {"image_1": nodes.LoadImage().load_image(name)[0], **references}
                positive, negative, latent = TextEncodeQwenImage21.execute(
                    clip, request["prompt"], "", vae, resolution=0, images=images
                ).result
                seed = secrets.randbits(64)
                samples = nodes.KSampler().sample(model, seed, 25, 1.0, "euler", "simple", positive, negative, latent, denoise=1.0)[0]
                decoded = nodes.VAEDecode().decode(vae, samples)[0]
                comfy.model_management.throw_exception_if_processing_interrupted()
                prefix = request["batch"] + "/edited"
                graph = workflow(request["prompt"], [name, *input_names["references"]], seed, prefix)
                save = SaveImageAdvanced.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.prompt: graph}})
                saved = save.execute(decoded, prefix, {"format": "png", "bit_depth": "8-bit", "input_color_space": "sRGB"}).ui["images"][0]
                generated = Path(runtime["output"]) / saved["subfolder"] / saved["filename"]
                emit({"type": "image", "file": source, "generated": str(generated)})
                del images, positive, negative, latent, samples, decoded
            emit({"type": "complete"})
    except comfy.model_management.InterruptProcessingException:
        emit({"type": "stopped"})


if __name__ == "__main__":
    try:
        run()
    except Exception:
        error = traceback.format_exc()
        sys.stderr.write(error)
        emit({"type": "error", "error": error})
        sys.exit(1)
