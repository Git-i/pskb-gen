TextureCube env_map : register(t0);
RWTexture2DArray<float4> pf_map : register(u0, space1);
SamplerState ss : register(s1);
struct roughness
{
    float value;
};
[[vk::push_constant]] roughness in_roughness : register(b2);
static const float PI = 3.142857;
float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H = {
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta,
    };

    // from tangent-space vector to world-space sample vector
    float3 up = float3(1.0, 0.0, 0.0);
    float3 tangent = cross(up, N);
    tangent = lerp(cross(N, float3(1.0, 0.0, 0.0)), tangent, step(0.000001, dot(tangent, tangent)));
    tangent = normalize(tangent);
    float3 bitangent = normalize(cross(N, tangent));
    float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}
float2 Hammersley(uint i, uint N)
{
    return float2(float(i)/float(N), RadicalInverse_VdC(i));
}
float3 GetSamplerVector(uint3 DTid)
{
    uint width, height, elem;
    pf_map.GetDimensions(width, height, elem);
    float2 st = DTid.xy / float2(width, height);
    float2 uv = 2.0 * float2(st.x, st.y) - 1.f.xx;
    switch (DTid.z)
    {
    case 0: return float3(1.0f, uv.y, -uv.x);
    case 1: return float3(-1.0f, uv.y, uv.x);
    case 2: return float3(uv.x, -1.0f, uv.y);
    case 3: return float3(uv.x, 1.0f, -uv.y);
    case 4: return float3(uv.x, uv.y, 1.0f);
    case 5: return float3(-uv.x, uv.y, -1.0f);
    default: return float3(0.0f, 0.0f, 0.0f);
    }
}
[numthreads(1,1,1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    float3 normal = normalize(GetSamplerVector(DTid));
    float3 R = normal;
    float3 V = normal;

    const uint sample_count = 1024;
    float4 prefilteredColor = 0.xxxx;
    float totalWeight = 0;
    for (uint i = 0; i < sample_count; i++)
    {
        float2 Xi = Hammersley(i, sample_count);
        float3 H = ImportanceSampleGGX(Xi, normal, in_roughness.value);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(normal, L), 0.0);
        if(NdotL > 0.0)
        {
            prefilteredColor += env_map.Sample(ss, L) * NdotL;
            totalWeight += NdotL;
        }
    }
    pf_map[DTid] = 	float4((prefilteredColor / totalWeight).rgb, 1.0);
}

