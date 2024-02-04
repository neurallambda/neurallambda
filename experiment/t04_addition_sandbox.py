'''.

Here, I'm trying to experiment with toy non-ML problems in python to understand
what needs to go into the ML version.


THOUGHTS

* Per conditional branch, stacks can be read once, popped once, and pushed
  once. Same for queues.

'''


##########
#

def print_grid(data, labels=None):
    data = list(data)  # Convert the data to a list if it's not already. Data should be iterable of iterables.
    column_widths = []  # This will store the maximum width needed for each column.
    max_columns = 0  # Stores the maximum number of columns in any row.

    # Calculate the max number of columns in any row
    for row in data:
        for column in row:
            max_columns = max(max_columns, len(str(column)))  # Update max_columns based on length of each column.

    # Initialize column widths to 0
    column_widths = [0] * max_columns  # Initialize column widths array with zeros based on max_columns.

    # Update column widths based on the data
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            for k, item in enumerate(column):
                if isinstance(item, float):
                    w = 6  # For floating point numbers, fix width to 6 (including decimal point and numbers after it).
                else:
                    w = len(str(item))  # For other types, set width to the length of item when converted to string.
                column_widths[k] = max(column_widths[k], w)  # Update column width if current item's width is larger.

    # Print the grid with aligned columns
    for row in data:
        if labels is None:
            labels = [''] * len(row)  # If no labels provided, create empty labels for alignment.
        max_label = max(map(len, labels))  # Find the maximum label width for alignment.
        for lab, column in zip(labels, row):
            print(f"{lab.rjust(max_label)}", end=" ")  # Print label right-justified based on max_label width.
            for i, item in enumerate(column):
                if isinstance(item, float):
                    x = f'{item:>.2f}'  # Format floating point numbers to 2 decimal places.
                else:
                    x = str(item)  # Convert other types to string.
                print(f"{x.rjust(column_widths[i])}", end=" ")  # Right-justify item based on column width.
            print()  # Newline after each column.
        print("-" * (sum(column_widths) + max_label + 1))  # Print separator line after each row.


##########
#

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) < 1:
            return None
        return self.stack.pop()

    def peek(self):
        if len(self.stack) < 1:
            return None
        return self.stack[-1]

class Queue:
    def __init__(self):
        self.queue = []

    def put(self, item):
        self.queue.append(item)

    def get(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def peek(self):
        if len(self.queue) < 1:
            return None
        return self.queue[0]


##########
#

print()
print('--------------------------------------------------')
print('SUM')
print()


def running_sum(inp, work_stack, work_queue, control_stack, control_queue):
    ''' (Neither queue is used in this particular game.)

    HOW TO CONVERT TO NEURALLAMBDA

    * Peek at everything

    * Project control and inp through symbol identifier

    * Calculate output (semantics): inp (id?), peeks (id?)

    * Calculate control (syntax): inp, out, peeks

    * Project symbol IDs through decision mode: inp, out, new control, peeks

    * Manipulate stacks/queues

    '''
    control = control_stack.peek()
    work = work_stack.peek()  # in this game, the stack holds `sum`

    if control == 'RETURN_ANSWER_OPEN':
        control_stack.pop()
        control_stack.push('RETURN_SUM')
        return '<answer>', work_stack, work_queue, control_stack, control_queue

    if control == 'RETURN_SUM':
        work_stack.pop()
        control_stack.pop()
        control_stack.push('RETURN_ANSWER_CLOSE')
        return work, work_stack, work_queue, control_stack, control_queue

    if control == 'RETURN_ANSWER_CLOSE':
        control_stack.pop()
        control_stack.push('NA')
        return '</answer>', work_stack, work_queue, control_stack, control_queue

    if isinstance(inp, int):
        work = work_stack.pop()
        work_stack.push(work + inp if work is not None else inp)
        return 'NA', work_stack, work_queue, control_stack, control_queue

    if inp == 'FINISHED':
        control_stack.pop()
        control_stack.push('RETURN_ANSWER_OPEN')
        return 'NA', work_stack, work_queue, control_stack, control_queue

    return 'NA', work_stack, work_queue, control_stack, control_queue

inps = [1, 2, 3, "FINISHED", "NA", "NA", "NA", "NA", "NA"]
exps = ["NA", "NA", "NA", "NA", "<answer>", 6, "</answer>", "NA"]

work_stack = Stack()
work_queue = Queue()
control_stack = Stack()
control_queue = Queue()

debug = []

for inp, exp in zip(inps, exps):
    out, work_stack, work_queue, control_stack, control_queue = running_sum(inp, work_stack, work_queue, control_stack, control_queue)
    debug.append([[out],
                  [exp],
                  [work_stack.peek()],
                  [work_queue.peek()],
                  [control_stack.peek()],
                  [control_queue.peek()],
                  ])

print_grid(debug, labels=['out', 'exp', 'WS', 'WQ', 'CS', 'CQ'])


##########
#

print()
print()
print()
print('--------------------------------------------------')
print('NBACK')
print()

def n_back(inp, work_stack, work_queue, control_stack, control_queue):
    control = control_stack.peek()
    work = work_stack.peek()

    if control == 'RETURN_ANSWER_OPEN':
        control_stack.pop()
        control_stack.push('RETURN_N_BACK')
        return '<answer>', work_stack, work_queue, control_stack, control_queue

    if control == 'RETURN_N_BACK':
        out = work_queue.get()
        control_stack.pop()
        control_stack.push('RETURN_ANSWER_CLOSE')
        return out, work_stack, work_queue, control_stack, control_queue

    if control == 'RETURN_ANSWER_CLOSE':
        control_stack.pop()
        control_stack.push('NA')
        return '</answer>', work_stack, work_queue, control_stack, control_queue

    if isinstance(inp, int):
        work_queue.get()
        work_queue.put(inp)
        return 'NA', work_stack, work_queue, control_stack, control_queue

    if inp == 'FINISHED':
        control_stack.pop()
        control_stack.push('RETURN_ANSWER_OPEN')
        return 'NA', work_stack, work_queue, control_stack, control_queue

    return 'NA', work_stack, work_queue, control_stack, control_queue

inps = [1, 2, 3, 4, "FINISHED", "NA", "NA", "NA", "NA"]
exps = ["NA", "NA", "NA", "NA", "NA", "<answer>", 3, "</answer>", "NA"]

work_stack = Stack()
work_queue = Queue()
control_stack = Stack()
control_queue = Queue()

# offset for 2 back
work_queue.put('NA')
work_queue.put('NA')

debug = []
debug2 = []
for inp, exp in zip(inps, exps):
    out, work_stack, work_queue, control_stack, control_queue = n_back(inp, work_stack, work_queue, control_stack, control_queue)
    debug2.append(out)
    debug.append([[out],
                  [exp],
                  [work_stack.peek()],
                  [work_queue.peek()],
                  [control_stack.peek()],
                  [control_queue.peek()],
                  ])


print_grid(debug, labels=['out', 'exp', 'WS', 'WQ', 'CS', 'CQ'])
print(', '.join([str(x) for x in debug2]))




##########
#

print()
print()
print()
print('--------------------------------------------------')
print('SORT')
print()

def sort_inputs(inp, work_stack, work_queue, control_stack, control_queue):
    control = control_stack.peek()

    if control == 'COLLECT_INPUTS':
        if isinstance(inp, int):
            work_queue.put(inp)  # Collecting inputs
            return 'NA', work_stack, work_queue, control_stack, control_queue
        elif inp == 'FINISHED':
            # Transition to sorting phase
            control_stack.pop()
            control_stack.push('SORTING_IN_PROGRESS')
            return 'NA', work_stack, work_queue, control_stack, control_queue

    elif control == 'SORTING_IN_PROGRESS':
        if not work_queue.peek() is None:
            # Begin or continue sorting
            temp = work_queue.get()
            while not work_stack.peek() is None and work_stack.peek() > temp:
                work_queue.put(work_stack.pop())
            work_stack.push(temp)
            while not work_queue.peek() is None and work_queue.peek() < work_stack.peek():
                work_stack.push(work_queue.get())
        else:
            control_stack.pop()
            control_stack.push('OUTPUT_SORTED')
        return 'NA', work_stack, work_queue, control_stack, control_queue

    elif control == 'OUTPUT_SORTED':
        if not work_stack.peek() is None:
            # Output sorted elements one by one
            return work_stack.pop(), work_stack, work_queue, control_stack, control_queue
        else:
            # Sorting and output are complete
            control_stack.pop()
            control_stack.push('NA')
            return 'NA', work_stack, work_queue, control_stack, control_queue

    return 'NA', work_stack, work_queue, control_stack, control_queue


inputs = [5, 3, 7, 2, 'FINISHED', 'NA', 'NA', 'NA', 'NA', 'NA']
expected_outputs = ['NA', 'NA', 'NA', 'NA', 'NA', 2, 3, 5, 7, 'NA']

work_stack = Stack()
work_queue = Queue()
control_stack = Stack()
control_queue = Queue()
control_stack.push('COLLECT_INPUTS')  # Initialize control stack

debug = []

for inp, exp in zip(inputs, expected_outputs):
    out, work_stack, work_queue, control_stack, control_queue = sort_inputs(inp, work_stack, work_queue, control_stack, control_queue)
    debug.append([out, exp, work_stack.stack, work_queue.queue, control_stack.stack, control_queue.queue])

for entry in debug:
    print(f"Output: {entry[0]}, Expected: {entry[1]}, Work Stack: {entry[2]}, Work Queue: {entry[3]}, Control Stack: {entry[4]}, Control Queue: {entry[5]}")



##################################################
#

def two_stack_sort_corrected(arr):
    input_stack = arr  # Direct use of input array as the input stack
    temp_stack = []

    while input_stack:
        current = input_stack.pop()
        while temp_stack and temp_stack[-1] < current:  # Ensure temp stack is in descending order
            input_stack.append(temp_stack.pop())
        temp_stack.append(current)

    # Since temp_stack is in descending order, reverse it to get the sorted array in ascending order
    sorted_arr = []
    while temp_stack:
        sorted_arr.append(temp_stack.pop())

    return sorted_arr

def non_recursive_quick_sort_corrected(arr):
    stack = [(0, len(arr) - 1)]

    while stack:
        start, end = stack.pop()
        if start >= end:
            continue

        pivot, left, right = arr[end], start, end - 1
        while left <= right:
            while left <= right and arr[left] < pivot:
                left += 1
            while left <= right and arr[right] > pivot:
                right -= 1
            if left <= right:
                arr[left], arr[right] = arr[right], arr[left]
                left, right = left + 1, right - 1

        arr[left], arr[end] = arr[end], arr[left]  # Place pivot in its correct position
        stack.append((start, left - 1))
        stack.append((left + 1, end))

    return arr

def iterative_merge_sort_with_queues_corrected(arr):
    if len(arr) <= 1:
        return arr

    queues = [[item] for item in arr]  # Initialize with each element in its own queue

    # Iteratively merge queues
    while len(queues) > 1:
        new_queues = []
        for i in range(0, len(queues), 2):
            if i + 1 < len(queues):
                merged_queue = []
                while queues[i] or queues[i + 1]:
                    if not queues[i]:
                        merged_queue.extend(queues[i + 1])
                        break
                    elif not queues[i + 1]:
                        merged_queue.extend(queues[i])
                        break
                    else:
                        if queues[i][0] <= queues[i + 1][0]:
                            merged_queue.append(queues[i].pop(0))
                        else:
                            merged_queue.append(queues[i + 1].pop(0))
                new_queues.append(merged_queue)
            else:
                new_queues.append(queues[i])
        queues = new_queues

    return queues[0]

# Example usage
arr = [3, 6, 8, 10, 1, 2, 1]
print("Corrected Two Stack Sort:", two_stack_sort_corrected(arr.copy()))
print("Corrected Non-Recursive Quick Sort:", non_recursive_quick_sort_corrected(arr.copy()))
print("Corrected Iterative Merge Sort with Queues:", iterative_merge_sort_with_queues_corrected(arr.copy()))



##################################################

def counting_sort(arr):
    max_element = max(arr)
    count_array_length = max_element + 1
    count_array = [0] * count_array_length

    # Count each element
    for num in arr:
        count_array[num] += 1

    # Accumulate counts
    for i in range(1, count_array_length):
        count_array[i] += count_array[i - 1]

    # Place the elements in output array
    output_array = [0] * len(arr)
    i = len(arr) - 1
    while i >= 0:
        current_element = arr[i]
        count_array[current_element] -= 1
        new_position = count_array[current_element]
        output_array[new_position] = current_element
        i -= 1

    return output_array

print('Counting sort:', counting_sort(arr))



##################################################

from collections import deque

def radix_sort(arr):
    max_length = len(str(max(arr)))  # Maximum number of digits
    queues = {i: deque() for i in range(10)}  # Queues for digits 0-9

    for position in range(max_length):
        for number in arr:
            # Finding the digit at the current position
            digit = (number // (10**position)) % 10
            queues[digit].append(number)

        # Collect numbers from queues in order
        arr = []
        for i in range(10):
            while queues[i]:
                arr.append(queues[i].popleft())

    return arr

print('Radix sort:', radix_sort(arr))



##################################################

def bucket_sort(arr):
    max_element = max(arr)
    min_element = min(arr)
    bucket_range = max_element - min_element + 1  # Range of bucket values
    buckets = [[] for _ in range(bucket_range)]  # Initialize buckets

    # Distribute array elements into buckets
    for num in arr:
        index = num - min_element  # Bucket index
        buckets[index].append(num)

    # Collect elements from buckets
    output_array = []
    for bucket in buckets:
        for num in bucket:
            output_array.append(num)

    return output_array


print('Bucket sort:', bucket_sort(arr))


##################################################

def pigeonhole_sort(arr):
    # Step 1: Find the range of the values (min and max)
    min_val = min(arr)
    max_val = max(arr)
    size = max_val - min_val + 1

    # Step 2: Create the pigeonhole array
    holes = [[] for _ in range(size)]

    # Step 3: Distribute the elements into pigeonholes
    for x in arr:
        assert isinstance(x, int), "Pigeonhole sort is for integers."
        holes[x - min_val].append(x)

    # Step 4: Collect the elements in order
    i = 0
    for hole in holes:
        for x in hole:
            arr[i] = x
            i += 1

    return arr

print("Pigeonhole:", pigeonhole_sort(arr))
