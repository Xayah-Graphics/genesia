include_guard(GLOBAL)

set(GENESIA_CLASSIFIER_INITIAL_MODEL "${GENESIA_ASSET_DIRECTORY}/classifier/weights/convnextv2-tiny.safetensors")
file(MAKE_DIRECTORY "${GENESIA_ASSET_DIRECTORY}/classifier/weights")
file(DOWNLOAD
        "https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k/resolve/b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817/model.safetensors"
        "${GENESIA_CLASSIFIER_INITIAL_MODEL}"
        EXPECTED_HASH SHA256=6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9
        TLS_VERIFY ON
        SHOW_PROGRESS
)
