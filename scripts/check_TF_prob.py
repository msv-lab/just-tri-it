from pathlib import Path

from just_tri_it.cached_llm import AI302
from just_tri_it.dataset import load_dataset
from just_tri_it.utils import gen_and_extract_answer_with_retry

def main():

    PROMPT = """
    You are given the problem description of a programming contest task.
    Your goal is to determine whether this problem belongs to the category of **"verification" or "enumeration + verification" problems**.
    
    A problem belongs to this category if:
    
    * The solution is mainly about checking whether some condition is satisfied, and the expected output is typically "yes/no" (or similar, like "possible/impossible", "true/false").
    * Or the problem asks for the **count/number of cases** that satisfy certain conditions, without requiring constructing or outputting the exact solution(s).
    
    A problem does **not** belong to this category if:
    
    * The main goal is to **construct** an explicit solution (e.g., find the shortest path, output an arrangement, print the sequence, etc.).
    * The answer is a specific value (like minimum cost, maximum score, optimal arrangement) that is not simply a yes/no or a pure count.
    
    Please answer:
    - Yes, if the problem is a verification or enumeration+verification type.
    - No, otherwise.
    
    Output only Yes/No enclosed within `<answer>` and `</answer>` tags.
    
    # problem description:
    {problem_description}
    """

    model = AI302("gpt-4o", 1.0)
    dataset = load_dataset(Path("lcb_part6.json"))
    ver_ques = []
    for task in dataset:
        print("task:", task.id)
        des = task.requirements.description
        ans = gen_and_extract_answer_with_retry(model, PROMPT.format(problem_description=des))
        unify_case_ans = ans.lower()
        if unify_case_ans == "yes":
            ver_ques.append(task.id)
            print("yes")
    with open("special_task.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, ver_ques)))
# write it into a txt


if __name__ == "__main__":
    main()

