import Lake

open System Lake DSL

package NumLeanOpenCL where
  version := v!"0.1.0"
  moreLinkArgs := #[
    "-Wl,--allow-shlib-undefined",
    "-Wl,-rpath,/home/tskrivan/.elan/toolchains/leanprover--lean4---v4.30.0/lib/lean",
    "/usr/lib/x86_64-linux-gnu/libOpenCL.so"
  ]

require "lecopivo" / NumLean from git "https://github.com/lecopivo/NumLean"

input_file opencl_lean_context.c where
  path := "c" / "opencl_lean_context.c"
  text := true

input_file opencl_lean_float32array.c where
  path := "c" / "opencl_lean_float32array.c"
  text := true

target opencl_lean_context.o pkg : FilePath := do
  let src ← opencl_lean_context.c.fetch
  buildO (pkg.buildDir / "c" / "opencl_lean_context.o") src
    #["-I", (← getLeanIncludeDir).toString] #["-fPIC"] "cc" getLeanTrace

target opencl_lean_float32array.o pkg : FilePath := do
  let src ← opencl_lean_float32array.c.fetch
  buildO (pkg.buildDir / "c" / "opencl_lean_float32array.o") src
    #["-I", (← getLeanIncludeDir).toString] #["-fPIC"] "cc" getLeanTrace

target liblean_numleanopencl_opencl pkg : Dynlib := do
  let libName := "lean_numleanopencl_opencl"
  let contextO ← opencl_lean_context.o.fetch
  let arrayO ← opencl_lean_float32array.o.fetch
  let weakArgs := #[
    s!"-Wl,-rpath,{(← getLeanLibDir).toString}",
    "/usr/lib/x86_64-linux-gnu/libOpenCL.so"
  ]
  buildLeanSharedLib libName (pkg.sharedLibDir / nameToSharedLib libName)
    #[contextO, arrayO] #[] weakArgs

lean_lib NumLeanOpenCL where
  precompileModules := true
  moreLinkLibs := #[liblean_numleanopencl_opencl]

@[default_target] lean_exe numleanopencl where root := `Main

lean_exe opencl_profile where root := `Profile

@[test_driver] lean_exe numleanopencl_tests where
  root := `Tests.Main
