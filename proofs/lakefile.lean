import Lake
open Lake DSL

package proofs

@[default_target]
lean_lib Proofs where
  -- 你的四个证明文件都放在 Proofs/ 目录下
  srcDir := "."


require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "master"
