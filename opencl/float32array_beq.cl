__kernel void numlean_opencl_float32array_beq(__global const float* a,
                                             __global const float* b,
                                             __global uint* out,
                                             ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n && a[gid] != b[gid]) {
        out[0] = 0;
    }
}
