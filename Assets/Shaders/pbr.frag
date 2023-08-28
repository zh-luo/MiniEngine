#version 330 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;
layout(location = 2) out int EntityID;

in vec2 TexCoords;
in vec3 normalWS;
in vec4 shadowCoord;
in vec3 fragPos;

uniform sampler2D   DiffuseMap;
uniform sampler2D   DepthMap;
uniform samplerCube IrradianceMap;
uniform samplerCube PrefilterMap;
uniform sampler2D   BRDF_LUT;
uniform sampler2D   Emu;

uniform int useEnvMap = 1;
uniform int kullaConty;
uniform int entityID;

uniform vec3  lightPosition;
uniform vec3  lightDirection;
uniform vec3  lightColor;
uniform float lightIntensity;

uniform float bloomThreshold;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 viewPos;

// material parameters
uniform float baseF;
uniform vec3  baseColor;
uniform float metallic;
uniform float roughness;
uniform float ao;

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

// magic number
#define VISIBLE_VALUE 1.0
#define INVISIBLE_VALUE 0.6
#define BIAS 0.005
#define RESOLUTION 2048.0
#define SEARCH_RADIUS 1024.0
#define LIGHT_WIDTH 1.0
#define USE_POISSON 1

#define MAX_REFLECTION_LOD 4.0

vec2 poissonDisk[NUM_SAMPLES];

float unpack(vec4 rgbaDepth)
{
    const vec4 bitShift = vec4(1.0, 1.0 / 256.0, 1.0 / (256.0 * 256.0), 1.0 / (256.0 * 256.0 * 256.0));
    return dot(rgbaDepth, bitShift);
}

highp float rand_2to1(vec2 uv)
{
    // 0 - 1
    const highp float a = 12.9898, b = 78.233, c = 43758.5453;
    highp float       dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, PI);
    return fract(sin(sn) * c);
}

void poissonDiskSamples(const in vec2 randomSeed)
{
    float ANGLE_STEP      = PI2 * float(NUM_RINGS) / float(NUM_SAMPLES);
    float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);

    float angle      = rand_2to1(randomSeed) * PI2;
    float radius     = INV_NUM_SAMPLES;
    float radiusStep = radius;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        poissonDisk[i] = vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
        radius += radiusStep;
        angle += ANGLE_STEP;
    }
}

float ShadowMapping(vec3 coords)
{
    float bias = max(0.05 * (dot(normalWS, lightDirection)), 0.005);

    if (coords.z > 1.0)
        return INVISIBLE_VALUE;
    float closestDepth = texture(DepthMap, coords.xy).r;
    float currentDepth = coords.z;
    float visibility   = currentDepth > closestDepth + bias ? INVISIBLE_VALUE : VISIBLE_VALUE;

    return visibility;
}

float PCF(vec4 coords, float filterRadius)
{
#ifdef USE_POISSON
    poissonDiskSamples(coords.xy);
#else
    uniformDiskSamples(coords.xy);
#endif

    float bias   = max(0.05 * (dot(normalWS, lightDirection)), 0.005);
    bias         = 0.01;
    float result = 0.0;

    for (int i = 0; i < PCF_NUM_SAMPLES; ++i) {
        // float depth = unpack(texture(DepthMap, coords.xy + poissonDisk[i] * filterRadius / RESOLUTION)) + BIAS;
        vec2  offset = poissonDisk[i] * filterRadius / RESOLUTION;
        float depth  = texture(DepthMap, coords.xy + offset).r + bias;
        result += depth > coords.z ? VISIBLE_VALUE : INVISIBLE_VALUE;
    }
    return result / float(PCF_NUM_SAMPLES);
}

float findBlocker(vec2 uv, float zReceiver)
{
#ifdef USE_POISSON
    poissonDiskSamples(uv);
#else
    uniformDiskSamples(uv);
#endif
    float biasBase = 0.2;
    float bias     = max(biasBase * (dot(normalWS, lightDirection)), biasBase);

    float blockerSum = 0.0;
    int   blockerCnt = 0;
    for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i) {
        float depth = texture(DepthMap, uv + poissonDisk[i] / RESOLUTION).r + bias;
        if (depth < zReceiver) {
            ++blockerCnt;
            blockerSum += depth;
        }
    }
    if (blockerCnt == 0)
        return 2.0;
    if (blockerCnt == BLOCKER_SEARCH_NUM_SAMPLES)
        return -1.0;
    return blockerSum / float(blockerCnt);
}

float PCSS(vec4 coords)
{
    // STEP 1: avgblocker depth
    float zBlocker = findBlocker(coords.xy, coords.z);
    // 没有遮挡
    if (abs(zBlocker - 2.0) < EPS)
        return VISIBLE_VALUE;
    // 完全遮挡
    if (abs(zBlocker - -1.0) < EPS)
        return INVISIBLE_VALUE;

    // STEP 2: penumbra size
    float penumbra = (coords.z - zBlocker) * LIGHT_WIDTH / zBlocker;
    // STEP 3: filtering
    return PCF(coords, penumbra);
    // return penumbra;
}

float CalculateShadow()
{
    vec3 coords = shadowCoord.xyz / shadowCoord.w;
    coords      = coords * 0.5 + 0.5;

    // return PCSS(vec4(coords, 1.0));
    return PCF(vec4(coords, 1.0), 2);
    // return ShadowMapping(coords);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom       = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float kIBL = (roughness * roughness) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) { return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0); }

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

//https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
vec3 AverageFresnel(vec3 r, vec3 g)
{
    return vec3(0.087237) + 0.0230685 * g - 0.0864902 * g * g + 0.0774594 * g * g * g + 0.782654 * r - 0.136432 * r * r +
           0.278708 * r * r * r + 0.19744 * g * r + 0.0360605 * g * g * r - 0.2586 * g * r * r;
}

vec3 MultiScatterBRDF(float NdotL, float NdotV)
{
    vec3 albedo = pow(texture(DiffuseMap, TexCoords).rgb, vec3(2.2));

    vec3 E_o   = texture(Emu, vec2(NdotL, roughness)).xyz;
    vec3 E_i   = texture(Emu, vec2(NdotV, roughness)).xyz;
    vec3 E_avg = texture(Emu, vec2(0, roughness)).xyz;
    E_o        = clamp(E_o, 0.0, 0.999);
    E_i        = clamp(E_i, 0.0, 0.999);
    E_avg      = clamp(E_avg, 0.0, 0.999);

    vec3 edgetint = vec3(0.827, 0.792, 0.678);
    vec3 F_avg    = AverageFresnel(albedo, edgetint);

    // TODO: To calculate fms and missing energy here
    vec3 fms  = (1.0 - E_o) * (1.0 - E_i) / (PI * (1.0 - E_avg));
    vec3 fadd = F_avg * E_avg / (1.0 - F_avg * (1.0 - E_avg));

    return fms * fadd;
}

void main()
{
    vec3 albedo = texture(DiffuseMap, TexCoords).rgb * baseColor;

    vec3 N = normalize(normalWS);
    vec3 V = normalize(viewPos - fragPos);

    vec3 F0 = vec3(0.04);
    F0      = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 pointLo = vec3(0.0);

    // =======================================================================================
    // Point Light
    for (int i = 0; i < 4; ++i) {
        // calculate per-light radiance
        vec3  L           = normalize(lightPositions[i] - fragPos);
        vec3  H           = normalize(V + L);
        float distance    = length(lightPositions[i] - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3  radiance    = lightColors[i] * attenuation;

        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3  F   = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);

        vec3  nominator   = NDF * G * F;
        float denominator = 4.0 * NdotV * NdotL;

        vec3 specular   = nominator / max(denominator, 0.001);
        vec3 lambertian = kD * albedo / PI;

        vec3 BRDF = lambertian + specular;

        vec3 Fms = MultiScatterBRDF(NdotL, NdotV);
        if (kullaConty == 1)
            BRDF += Fms;

        pointLo += BRDF * radiance * NdotL;
    }

    // =======================================================================================
    // Directional Light
    vec3 Lo = vec3(0.0);
    {
        vec3 L = normalize(lightPosition - fragPos);
        vec3 H = normalize(V + L);
        // float distance    = length(lightPosition - fragPos);
        // float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor;

        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3  F   = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        float NdotL = max(0, dot(N, L));
        float NdotV = max(0, dot(N, V));

        vec3  nominator   = NDF * G * F;
        float denominator = 4.0 * NdotV * NdotV;

        vec3 specular   = nominator / max(denominator, 0.001);
        vec3 lambertian = kD * albedo / PI;

        vec3 BRDF = lambertian + specular;

        vec3 Fms = MultiScatterBRDF(NdotL, NdotV);

        if (kullaConty == 1)
            BRDF += Fms;

        Lo += BRDF * radiance * NdotL;
        // add to outgoing radiance Lo
    }
    // =======================================================================================

    // ambient lighting (we now use IBL as the ambient term)
    vec3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 irradiance = texture(IrradianceMap, N).rgb;
    vec3 diffuse    = irradiance * albedo;

    vec3 L = normalize(lightPosition - fragPos);
    vec3 H = normalize(V + L);
    vec3 F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);
    vec3 R = reflect(-V, N);

    float NdotL = max(0, dot(N, L));
    float NdotV = max(0, dot(N, V));

    vec3 prefilteredColor = textureLod(PrefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;

    // specular when VdotH and roughness
    vec2 envBRDF  = texture(BRDF_LUT, vec2(NdotV, roughness)).rg;  // precompute BRDF = D * G * F / ...
    vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);
    vec3 BRDF     = kD * diffuse + specular;

    vec3 Fms = MultiScatterBRDF(NdotL, NdotV);

    vec3 ambient = (useEnvMap == 1 ? (BRDF) : vec3(0.03)) * ao;

    float visibility = CalculateShadow();
    vec3  color      = ((ambient + Lo + pointLo) * visibility) * albedo.rgb;
    FragColor        = vec4(color, 1.0);

    // Multi-target
    {
        float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
        if (brightness > bloomThreshold)
            BrightColor = vec4(FragColor.rgb, 1.0);

        EntityID = entityID;
    }
}
