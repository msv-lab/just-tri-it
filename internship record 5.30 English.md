# internship record （English）5.30

1. ## abstract

   the idea：resonate and predicate

   Objective: Based on a requirement R (which may be precise or imprecise, where "imprecise" means multiple valid outputs are allowed for a single input, distinct from a non-injective function; e.g., for the input set {1, 3, -2}, the requirement is to output a number greater than 0, where valid outputs could be 1 or 3), input a prompt that is as accurate and complete as possible to a large language model (LLM) to generate a target program (essentially code implementing the requirement). We must then verify the correctness of this generated program.

   Previous Work: We proposed methods to verify correctness, including:

   1.Consistency-based verification (maximum equivalence class partitioning),
   2.LLM self-reported confidence scores (weak correlation between confidence and correctness),
   3.Assertion function generation (assertions likely to share homologous errors with the target program).
   However, these methods fail to address the following issues:

   1.Homologous errors: All candidate programs within the same equivalence class may share the same logical flaws (e.g., misdefining critical conditions like x∈{1,2,3}). Assertions generated using the same flawed logic will falsely validate incorrect programs as "correct."
   2.Infinite input sets: Verification becomes impossible for requirements with infinite input domains (e.g., x∈Z) due to the inability to exhaustively test all inputs.
   3.Imprecise requirements: For ambiguous requirements (e.g., "output must be sorted," "no duplicates," "concise," or "add items"), the lack of executable, strict criteria makes it impossible to determine whether the output truly satisfies the semantics of the requirement. Without formal specifications (e.g., predicates or invariants), we cannot define what constitutes a "correct" output.

2. ## what is resonate

   First, we have a requirement R (here focusing solely on addressing homologous errors, temporarily disregarding precision). We instruct a large language model (LLM) to generate a target program based on R. We aim to use this method to determine whether the target program is correct while avoiding misjudgment caused by homologous errors.

   The target program generates an output z for every input x and y (generalized here; in reality, there may be multiple parameters beyond x and y). Additionally, we generate a family of resonators (not a single resonator, but multiple) based on the inverse requirement R ^−1 . Each resonator employs distinct algorithms and alters the semantics for the inverse requirement (this step is key to resolving homologous errors).

   Specifically, for the output Z obtained from the previous step and a sample input y, the resonators generate the input x (achieving partial inversion of parameters). We then define a resonance property: if satisfied, it strongly indicates the correctness of the target program (since the likelihood of homologous errors is now minimal). If not satisfied, we can rapidly localize the error.

   During the process of concretization, we encountered the following issues:

   1. For requirement R：There exists a distinction between precise and imprecise requirements. For imprecise requirements, fundamentally, a single input may correspond to multiple valid outputs. For example, consider the resonance property:

      ```
                                    f(r(f(x, y), y),y) ≈ f(x, y)
      ```

      When an input (x,y) maps to multiple outputs z1 ,z2 ,z3 , this property may no longer hold. However, this does not necessarily indicate that the program is incorrect.

   2. For infinite input domains：There are two scenarios:
      First: The input set itself is inherently infinite (e.g., x∈Z).
      Second: When constructing the resonator family based on the inverse requirement R ^−1 , the inverse process yields infinite outputs.For the first case, the inability to exhaustively test all inputs makes the verification of the property incomplete, undermining the credibility of the validation.
      For the second case, the inverse function r cannot reconstruct all possible original inputs x. The large language model (LLM) may autonomously compromise—for instance, truncating values or applying rounding—to produce results that appear correct but distort the property’s validity.

3. ## what is predicate

   R：Each user type corresponds to a pricing rule. User types are infinite. For user types 1, 2, and 3, the price remains unchanged. For all other user types, the price is increased by 10.

   target programme：

   ```
def f(x, y):
       if x in [1, 2, 3]:
           return y
       else:
           return y + 10
   ```
   
   predicate:

   ```
                         P = (x∈{1,2,3} ∧ z=y) ∨ (x∉{1,2,3} ∧ z=y+10)
   ```
   
   Formalization in Lean4:

   ```
def P' (x y z : Int) : Prop :=
     (x = 1 ∨ x = 2 ∨ x = 3) ∧ z = y ∨ -- x ∈ {1,2,3} → z = y
     (x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3) ∧ z = y + 10 -- x ∉ {1,2,3} → z = y+10
   
   ```
   
   The predicate P is derived from requirement R using logical expressions (equations, inequalities, set relations). Its purpose is to formally define the input-output constraints of the program. These constraints can be written and proven in formal languages like Lean4.

4. ##  predicate solve infinite input domains

   

5. ## predicate solve inexact requirements

   ## 

8. ##  example

   

   

