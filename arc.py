class Arc:
    def __init__(self, head : int, tail : int, weight,
                 head_node, tail_node):
        self.head = head
        self.head_node = head_node
        self.tail_node = tail_node
        self.tail = tail
        self.weight = weight

    def __repr__(self):
        return f"<v:{self.head}, u:{self.tail}, w:{self.weight}>"

    def __eq__(self, other):
        return self.head == other.head and self.tail == other.tail and \
               self.weight == other.weight