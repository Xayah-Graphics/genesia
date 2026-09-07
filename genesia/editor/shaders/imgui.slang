module imgui;

struct ImGuiVertex {
    float2 position;
    float2 texture_coordinate;
    uint color;
}

struct ImGuiPushData {
    DescriptorHandle<StructuredBuffer<ImGuiVertex>> vertices;
    DescriptorHandle<StructuredBuffer<uint>> indices;
    DescriptorHandle<Texture2D<float4>> texture;
    DescriptorHandle<SamplerState> sampler;
    uint index_offset;
    uint vertex_offset;
    float2 scale;
    float2 translation;
}

[vk::push_constant]
uniform ImGuiPushData push_data;

struct ImGuiVertexOutput {
    float4 position : SV_Position;
    float2 texture_coordinate : TEXCOORD0;
    float4 color : COLOR0;
}

[shader("vertex")]
ImGuiVertexOutput imgui_vertex(uint vertex_index : SV_VertexID) {
    StructuredBuffer<uint> indices = push_data.indices;
    StructuredBuffer<ImGuiVertex> vertices = push_data.vertices;
    ImGuiVertex vertex = vertices[push_data.vertex_offset + indices[push_data.index_offset + vertex_index]];
    ImGuiVertexOutput output;
    output.position = float4(vertex.position * push_data.scale + push_data.translation, 0.0f, 1.0f);
    output.texture_coordinate = vertex.texture_coordinate;
    output.color = float4(
        float(vertex.color & 0xffu),
        float((vertex.color >> 8u) & 0xffu),
        float((vertex.color >> 16u) & 0xffu),
        float((vertex.color >> 24u) & 0xffu)) / 255.0f;
    return output;
}

[shader("fragment")]
float4 imgui_fragment(ImGuiVertexOutput input) : SV_Target0 {
    Texture2D<float4> texture = push_data.texture;
    SamplerState sampler = push_data.sampler;
    float3 linear_color = select(input.color.rgb <= 0.04045f, input.color.rgb / 12.92f, pow((input.color.rgb + 0.055f) / 1.055f, 2.4f));
    return float4(linear_color, input.color.a) * texture.SampleLevel(sampler, input.texture_coordinate, 0.0f);
}
