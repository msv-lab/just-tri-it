import Lake
open Lake DSL

package proofs

@[default_target]
lean_lib Proofs where
  srcDir := "."


require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "master"
