TextureCube env_map;
RWTexture2DArray<float4> ir_map;
SamplerState ss;
static const float PI = 3.142857;

float3 GetSamplerVector(uint3 DTid)
{
    uint width, height, elem;
    ir_map.GetDimensions(width, height, elem);
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
void main(uint3 gid : SV_DispatchThreadID)
{
    float3 normal = normalize(GetSamplerVector(gid));
    float3 up = float3(0.0, 1.0, 0.0);
    float3 right = cross(normal, up);
    right = lerp(cross(normal, float3(1.0, 0.0, 0.0)), right, step(0.00001, dot(right, right)));

    right = normalize(right);
    up = normalize(cross(normal, right));
    const float sampleDelta = 0.025;

    float3 irradiance = 0.xxx;
    float nrSamples = 0;
    for (float phi = 0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for (float theta = 0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            float3 tangentSample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            float3 sampleVec = tangentSample.x * right + tangentSample.y * up +
                             tangentSample.z * normal;
            irradiance += env_map.SampleLevel(ss, sampleVec, 0).rgb * cos(theta) *
                          sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / nrSamples);
    ir_map[gid] = float4(irradiance, 1.0    );
}