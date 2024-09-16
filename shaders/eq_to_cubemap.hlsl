RWTexture2DArray<float4> outCubeMap;
Texture2D inTexture;
SamplerState ss;
static const float2 invAtan = float2(0.1591, 0.3183);
float2 SampleSphericalMap(float3 v)
{
    float2 uv = float2(atan2(v.x, v.z), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}
float3 GetSamplerVector(uint3 DTid)
{
    uint width, height, elem;
    outCubeMap.GetDimensions(width, height, elem);
    float2 st = DTid.xy/float2(width, height);
    float2 uv = 2.0 * float2(st.x, st.y) - 1.f.xx;
    switch(DTid.z)
    {
        case 0: return float3( 1.0f,  uv.y, -uv.x);
        case 1: return float3(-1.0f,  uv.y,  uv.x);
        case 2: return float3( uv.x, -1.0f,  uv.y);
        case 3: return float3( uv.x,  1.0f, -uv.y);
        case 4: return float3( uv.x,  uv.y,  1.0f);
        case 5: return float3(-uv.x,  uv.y, -1.0f);
        default: return float3(0.0f, 0.0f, 0.0f);
    }
}
[numthreads(1,1,1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    float3 transformed = normalize(GetSamplerVector(DTid));
    outCubeMap[DTid] = inTexture.SampleLevel(ss,SampleSphericalMap(transformed),0);
}   
