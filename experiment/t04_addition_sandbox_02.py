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


def running_sum(inp, work, control):
    def control_stack_decision(control, inp):
        ''' push/pop/null_op refer to what should be the final action of the whole function '''
        match control:
            case 'RETURN_ANSWER_OPEN':
                return ('PUSH', 'RETURN_SUM')
            case 'RETURN_SUM':
                return ('PUSH', 'RETURN_ANSWER_CLOSE')
            case 'RETURN_ANSWER_CLOSE':
                return ('PUSH', 'N')
            case _ if isinstance(inp, int):
                return ('NULL_OP', None)
            case _ if inp == 'F':
                return ('PUSH', 'RETURN_ANSWER_OPEN')
            case _:
                return ('NULL_OP', None)

    def work_stack_decision(control, inp):
        ''' push/pop/null_op refer to what should be the final action of the whole function '''
        match control:
            case 'RETURN_ANSWER_OPEN' | 'RETURN_ANSWER_CLOSE':
                return 'NULL_OP'
            case 'RETURN_SUM':
                return 'POP'
            case _ if isinstance(inp, int):
                return 'PUSH'
            case _:
                return 'NULL_OP'

    def work_stack_semantics(work, inp):
        if isinstance(inp, int):
            return work + inp if work is not None else inp
        return None

    def out_computation(control, work):
        # Output decision based on control state
        match control:
            case 'RETURN_ANSWER_OPEN':
                return 'L'
            case 'RETURN_SUM':
                return work
            case 'RETURN_ANSWER_CLOSE':
                return 'R'
            case _:
                return 'N'

    work_val = work_stack_semantics(work, inp)
    control_op, control_val = control_stack_decision(control, inp)
    work_op = work_stack_decision(control, inp)
    out = out_computation(control, work)

    return (
        out,
        work_op, work_val,
        control_op, control_val
    )

def apply_stack_op(stack, op, old_val, new_val):
    '''WARN: this is weird. It expects that a pop was already performed, so
       operations are shifted by one. IE if you say POP, it will recognize that
       one was already performed, so don't do anything. '''
    if op == 'NULL_OP':
        stack.push(old_val)  # Restore if no operation was performed
    elif op == 'PUSH':
        stack.push(new_val)
    elif op == 'POP':
        pass # the original `pop` stands

inps = ['O', 'P', 'S',  1,   2,   3,  'T', 'T', "F", "N", "N", "N", 'N']
exps = ['N', 'N', 'N', "N", "N", "N", 'N', 'N', 'N', "L",  6,  "R", 'N']

work_stack = Stack()
work_queue = Queue()
control_stack = Stack()
control_queue = Queue()

debug = []
debug2 = []
for inp, exp in zip(inps, exps):
    work = work_stack.pop()
    control = control_stack.pop()

    out, work_op, new_work_val, control_op, new_control_val = running_sum(inp, work, control)

    apply_stack_op(work_stack, work_op, work, new_work_val)
    apply_stack_op(control_stack, control_op, control, new_control_val)

    debug2.append(str(out))
    debug.append([[out],
                  [exp],
                  [work_stack.peek()],
                  [work_queue.peek()],
                  [control_stack.peek()],
                  [control_queue.peek()],
                  ])

print_grid(debug, labels=['out', 'exp', 'WS', 'WQ', 'CS', 'CQ'])
print(', '.join(debug2))
