import Lake

open System Lake DSL

package NumLeanOpenCL where
  version := v!"0.1.0"
  moreLinkArgs := #["-Wl,--allow-shlib-undefined", "/usr/lib/x86_64-linux-gnu/libOpenCL.so"]

require "lecopivo" / NumLean from git "https://github.com/lecopivo/NumLean"

lean_lib NumLeanOpenCL

extern_lib liblean_numleanopencl_opencl (pkg) := do
  let contextSrcJob ← (inputFile (pkg.dir / "c" / "opencl_lean_context.c") true)
  let arraySrcJob ← (inputFile (pkg.dir / "c" / "opencl_lean_float32array.c") true)
  let lean ← getLeanInstall
  let contextOJob ← buildO
    (pkg.buildDir / "c" / "opencl_lean_context.o")
    contextSrcJob
    #["-I", lean.includeDir.toString]
    #["-fPIC"]
  let arrayOJob ← buildO
    (pkg.buildDir / "c" / "opencl_lean_float32array.o")
    arraySrcJob
    #["-I", lean.includeDir.toString]
    #["-fPIC"]
  buildStaticLib
    (pkg.staticLibDir / nameToStaticLib "lean_numleanopencl_opencl")
    #[contextOJob, arrayOJob]

@[default_target] lean_exe numleanopencl where root := `Main

lean_exe opencl_profile where root := `Profile

@[test_driver] lean_exe numleanopencl_tests where
  root := `Tests.Main
