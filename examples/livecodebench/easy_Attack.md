There is an enemy with stamina A.  Every time you attack the enemy, its stamina reduces by B.
At least how many times do you need to attack the enemy to make its stamina 0 or less?

Input

The input is given from Standard Input in the following format:
A B

Output

Print the answer.

Constraints


- 1 \le A,B \le 10^{18}
- A and B are integers.

Sample Input 1

7 3

Sample Output 1

3

Attacking three times make the enemy's stamina -2.
Attacking only twice makes the stamina 1, so you need to attack it three times.

Sample Input 2

123456789123456789 987654321

Sample Output 2

124999999

Sample Input 3

999999999999999998 2

Sample Output 3

499999999999999999
