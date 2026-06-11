__kernel void numlean_opencl_float32array_sum(__global const float* xs,
                                             __global float* out,
                                             ulong n) {
    float acc = 0.0f;
    for (ulong i = 0; i < n; i++) {
        acc += xs[i];
    }
    out[0] = acc;
}
