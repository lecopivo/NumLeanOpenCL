__kernel void numlean_opencl_float32array_add(__global float* a,
                                             __global const float* b,
                                             ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) {
        a[gid] += b[gid];
    }
}
