__kernel void smoke_add(__global const float* a,
                        __global const float* b,
                        __global float* out,
                        ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
