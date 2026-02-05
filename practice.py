'''
Array - Hash
'''
from typing import List

from matplotlib import collections

def containsDuplicate(nums: List[int]) -> bool:
    hashset = set()

    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False

def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    
    countS, countT = {}, {}

    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)
    return countS == countT

def twoSum(nums: List[int], target: int) -> List[int]:
    prevMap = {}

    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    ans = collections.defaultdict(list)

    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        ans[tuple(count)].append(s)
    return ans.values()

def topKFrequent(nums: List[int], k: int) -> List[int]:
    count = {}
    freq = [[] for i in range(len(nums) + 1)]

    for n in nums:
        count[n] = 1 + count.get(n, 0)

    for n, c in count.items():
        freq[c].append(n)

    res = []

    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
            
def productExceptSelf(nums: List[int]) -> List[int]:
    res = [1] * len(nums)

    prefix = 1
    for i in range(len(nums)):
        res[i] = prefix
        prefix *= nums[i]
    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
    return res

def isValidSudoku(board: List[List[str]]) -> bool:
    cols = collections.defaultdict(set)
    rows = collections.defaultdict(set)
    squares = collections.defaultdict(set)

    for r in range(9):
        for c in range(9):
            if board[r][c] == ".":
                continue
            if (
                board[r][c] in rows[r]
                or board[r][c] in cols[c]
                or board[r][c] in squares[(r // 3, c // 3)]
            ):
                return False
            cols[c].add(board[r][c])
            rows[r].add(board[r][c])
            squares[(r // 3, c // 3)].add(board[r][c])
    return True

def longestConsecutive(nums: List[int]) -> int:
    numSet = set(nums)
    longest = 0

    for n in nums:
        if (n - 1) not in numSet:
            length = 1
            while (n + length) in numSet:
                length += 1
            longest = max(longest, length)
    return longest

'''
Two Pointers
'''
def alphanum(c: str) -> bool:
    return (
        ord("A") <= ord(c) <= ord("Z")
        or ord("a") <= ord(c) <= ord("z")
        or ord("0") <= ord(c) <= ord("9")
    )

def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not alphanum(s[left]):
            left += 1
        while left < right and not alphanum(s[right]):
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return False

def twoSumII(numbers: List[int], target: int) -> List[int]:
    left, right = 0, len(numbers) - 1

    while left < right:
        curSum = numbers[left] + numbers[right]

        if curSum > target:
            right -= 1
        elif curSum < target:
            left += 1
        else:
            return [left + 1, right + 1]
        
def threeSum(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    for i, a in enumerate(nums):
        if a > 0:
            break

        if i > 0 and a == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            threeSum = a + nums[left] + nums[right]
            if threeSum > 0:
                right -= 1
            elif threeSum < 0:
                left += 1
            else:
                res.append([a, nums[left], nums[right]])
                left += 1
                right -= 1
                while nums[left] == nums[left - 1] and left < right:
                    left += 1
    return res

def trapRain(height: List[int]) -> int:
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    leftMax, rightMax = height[left], height[right]
    res = 0
    
    while left < right:
        if leftMax < rightMax:
            left += 1
            leftMax = max(leftMax, height[left])
            res += leftMax - height[left]
        else:
            right -= 1
            rightMax = max(rightMax, height[right])
            res += rightMax - height[right]
    return res

'''
Stack
'''

def isValidParentheses(s: str) -> bool:
    Map = {")": "(", "}": "{", "]": "["}
    stack = []

    for c in s:
        if c not in Map:
            stack.append(c)
            continue
        if not stack or stack[-1] != Map[c]:
            return False
        stack.pop()
    return not stack

class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.minStack[-1]
    
def evalPRN(tokens: List[str]) -> int:
    stack = []
    for c in tokens:
        if c == "+":
            stack.append(stack.pop() + stack.pop())
        elif c == "-":
            a, b = stack.pop(), stack.pop()
            stack.append(b - a)
        elif c == "*":
            stack.append(stack.pop() * stack.pop())
        elif c == "/":
            a, b = stack.pop(), stack.pop()
            stack.append(int(b / a))
        else:
            stack.append(int(c))
    return stack[0]


def generateParenthesis(n: int) -> List[str]:
    stack = []
    res = []

    def backtrack(openN: int, closeN: int) -> None:
        if openN == closeN == n:
            res.append("".join(stack))
            return
        
        if openN < n:
            stack.append("(")
            backtrack(openN + 1, closeN)
            stack.pop()
        
        if closeN < openN:
            stack.appned(")")
            backtrack(openN, closeN + 1)
            stack.pop()

    backtrack(0,0)
    return res

def dailyTemperatures(temps: List[int]) -> List[int]:
    res = [0] * len(temps)
    stack = []

    for i, t in enumerate(temps):
        while stack and t > stack[-1][0]:
            stackT, stackI = stack.pop()
            res[stackI] = i - stackI
        stack.append((t, i))
    return res


def carFleet(target: int, position: List[int], speed: List[int]) -> int:
    pair = [(pos, spd) for pos, spd in zip(position, speed)]
    pair.sort(reverse=True)
    stack = []

    for pos, spd in pair:
        stack.append((target - pos) / spd)
        if len(stack) >= 2 and stack[-1] <= stack[-2]:
            stack.pop()
    return len(stack)

def largestRectangleArea(heights: List[int]) -> int:
    maxArea = 0
    stack = []

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            maxArea = max(maxArea, height * (i - index))
            start = index
        stack.append((start, h))

    for i, h in stack:
        maxArea = max(maxArea, h * (len(heights) - i))
    return maxArea

"""
Sliding Window
"""

def maxProfit(prices: List[int]) -> int:
    res = 0
    lowest = prices[0]

    for price in prices:
        if price < lowest:
            lowest = price
        res = max(res, price - lowest)
    return res

def lengthOfLongestSubstring(s: str) -> int:
    charSet = set()
    left = 0
    res = 0

    for right in range(len(s)):
        while s[right] in charSet:
            charSet.remove(s[left])
            left += 1
        charSet.add(s[right])
        res = max(res, right - left + 1)
    return res

def characterReplacement(s: str, k: int) -> int:
    count = {}
    left = 0
    maxCount = 0

    for right in range(len(s)):
        count[s[right]] = 1 + count.get(s[right], 0)
        maxCount = max(maxCount, count[s[right]])

        if (right - left + 1) - maxCount > k:
            count[s[left]] -= 1
            left += 1
    
    return right - left + 1


def checkInclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False
    
    s1Count, s2Count = [0] * 26, [0] * 26
    for i in range(len(s1)):
        s1Count[ord(s1[i]) - ord("a")] += 1
        s2Count[ord(s2[i]) - ord("a")] += 1
    
    matches = 0
    for i in range(26):
        matches += 1 if s1Count[i] == s2Count[i] else 0
    
    left = 0
    for right in range(len(s1), len(s2)):
        if matches == 26:
            return True
        
        index = ord(s2[right]) - ord("a")
        s2Count[index] += 1
        if s1Count[index] == s2Count[index]:
            matches += 1
        elif s1Count[index] + 1 == s2Count[index]:
            matches -= 1

        index = ord(s2[left]) - ord("a")
        s2Count[index] -= 1
        if s1Count[index] == s2Count[index]:
            matches += 1
        elif s1Count[index] -1 == s2Count[index]:
            matches -= 1
        left += 1
    return matches == 26

def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    output = []
    q = collections.deque()
    left = right = 0

    while right < len(nums):
        while q and nums[q[-1]] < nums[right]:
            q.pop()
        q.append(right)

        if left > q[0]:
            q.popleft()

        if (right + 1) >= k:
            output.append(nums[q[0]])
            left += 1
        right += 1
    return output

def minWindow(s: str, t: str) -> str:
    if t == "":
        return ""
    
    countT, window = {}, {}
    for c in t:
        countT[c] = 1 + countT.get(c, 0)

    have, need = 0, len(countT)
    res, resLen = [-1, -1], float("infinity")

    left = 0
    for right in range(len(s)):
        c = s[right]
        window[c] = 1 + window.get(c, 0)

        if c in countT and window[c] == countT[c]:
            have += 1

        while have == need:
            if (right - left + 1) < resLen:
                res = [left, right]
                resLen = right - left + 1
            
            window[s[left]] -= 1
            if s[left] in countT and window[s[left]] < countT[s[left]]:
                have -= 1
            left += 1
    left , right = res
    return s[left : right + 1] if resLen != float("infinity") else ""


'''
Linked List
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def reverseList(head: ListNode) -> ListNode:
    prev, curr = None, head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev

def mergeTwoList(list1: ListNode, list2: ListNode) -> ListNode:
    dummy = ListNode()
    tail = dummy

    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    
    if list1:
        tail.next = list1
    elif list2:
        tail.next = list2
    
    return dummy.next

def hasCycle(head: ListNode) -> bool:
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def reorderList(head: ListNode) -> None:
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    second = slow.next
    prev = slow.next = None
    while second:
        temp = second.next
        second.next = prev
        prev = second
        second = temp

    first, second = head, prev
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    left = dummy
    right = head

    while n > 0:
        right = right.next
        n-= 1

    while right:
        left = left.next
        right = right.next
    
    left.next = left.next.next
    return dummy.next