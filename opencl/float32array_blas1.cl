__kernel void numlean_opencl_blas1_scal(__global float* x, float alpha, ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) x[gid] *= alpha;
}

__kernel void numlean_opencl_blas1_axpy(__global const float* x, __global float* y, float alpha, ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) y[gid] += alpha * x[gid];
}

__kernel void numlean_opencl_blas1_swap(__global float* x, __global float* y, ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) {
        float t = x[gid];
        x[gid] = y[gid];
        y[gid] = t;
    }
}

__kernel void numlean_opencl_blas1_rot(__global float* x, __global float* y, float c, float s, ulong n) {
    size_t gid = get_global_id(0);
    if (gid < n) {
        float xi = x[gid];
        float yi = y[gid];
        x[gid] = c * xi + s * yi;
        y[gid] = c * yi - s * xi;
    }
}

__kernel void numlean_opencl_blas1_dot(__global const float* x, __global const float* y, __global float* out, ulong n) {
    float acc = 0.0f;
    for (ulong i = 0; i < n; i++) acc += x[i] * y[i];
    out[0] = acc;
}

__kernel void numlean_opencl_blas1_asum(__global const float* x, __global float* out, ulong n) {
    float acc = 0.0f;
    for (ulong i = 0; i < n; i++) acc += fabs(x[i]);
    out[0] = acc;
}

__kernel void numlean_opencl_blas1_nrm2(__global const float* x, __global float* out, ulong n) {
    float acc = 0.0f;
    for (ulong i = 0; i < n; i++) acc += x[i] * x[i];
    out[0] = sqrt(acc);
}
