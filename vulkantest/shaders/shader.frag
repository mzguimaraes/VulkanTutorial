#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
//    vec3 playColor = fragColor;
//    playColor.x = fragColor.x < 0.0f ? 1.0f : fragColor.x - 0.1f;
//    playColor.y = fragColor.y < 0.0f ? 1.0f : fragColor.y - 0.1f;
//    playColor.z = fragColor.z < 0.0f ? 1.0f : fragColor.z - 0.1f;
    outColor = vec4(fragColor, 1.0);
    outColor = ubo.proj * ubo.view * ubo.model * outColor;
}
