extern "C" __global__ void strengths(
    const float* q, const int* edges, const float* observed,
    float* output, unsigned int* counts, int E, int T, int mode, int B) {
    int lane = threadIdx.x & 31;
    int edge = blockIdx.x * 8 + (threadIdx.x >> 5);
    int permutation = blockIdx.y;
    if (edge >= E || permutation >= B) return;
    int a = edges[3*edge], b = edges[3*edge+1], c = edges[3*edge+2];
    const float* row = q + (long long)permutation * 90 * T;
    float sum = 0.f, sumsq = 0.f;
    for (int t = lane; t < T; t += 32) {
        float v = row[a*T+t] * row[b*T+t];
        if (c >= 0) v *= row[c*T+t];
        sum += v;
        sumsq += v*v;
    }
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
    float mean = __shfl_sync(0xffffffff, sum, 0) / T;
    float variance = __shfl_sync(0xffffffff, sumsq, 0) / T - mean*mean;
    float total = 0.f;
    for (int t = lane; t < T; t += 32) {
        float va = row[a*T+t], vb = row[b*T+t];
        float product = va*vb;
        bool positive = va >= 0.f && vb >= 0.f;
        bool negative = va <= 0.f && vb <= 0.f;
        if (c >= 0) {
            float vc = row[c*T+t];
            product *= vc;
            positive = positive && vc >= 0.f;
            negative = negative && vc <= 0.f;
        }
        if (positive || negative) total += fabsf(product-mean);
    }
    for (int offset = 16; offset; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0) {
        float strength = variance > 0.f ? total/(T*sqrtf(variance)) : 0.f;
        if (mode == 0) output[(long long)permutation*E+edge] = strength;
        else if (strength >= observed[edge]) atomicAdd(counts+edge, 1u);
    }
}
