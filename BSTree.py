import queue
from typing import TypeVar, Generator, List, Tuple, Optional

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")      # represents a custom type used in application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height", "key"]

    def __init__(self, key: T, value: T = None, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value if value is not None else key
        self.parent, self.left, self.right = parent, left, right
        self.height = 0
        self.key = key

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class BSTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["root", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.root = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.root is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.root
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.root.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.root.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.root.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.root.height}".center(spaces)
        result += "\n\n"
        for i in range(self.root.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.root:
            self.root = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a right rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull left child up and shift left-right child across tree, update parent
        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        # left child has been pulled up to new root -> push old root down right, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.root:
            self.root = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def insert(self, root: Node, val: T) -> Node:
        """
        Insert a node with `val` into the subtree rooted at `root`.
        If `val` already exists in the AVL tree, do nothing.
        Return the root node of the balanced subtree following insertion.

        :param root: root node of subtree in which to insert.
        :param val: value to be inserted in subtree.
        :return: root node of balanced subtree.
        """
        # handle empty tree case
        if root is None:
            self.root = Node(val)
            self.size = 1
            return self.root

        # implement generic BST insertion
        if root.value == val:
            return root
        if val < root.value:
            # insert into left child or left subtree of `root`
            if root.left is None:
                root.left = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.left, val)
        else:
            # insert into right child or right subtree of `root`
            if root.right is None:
                root.right = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.right, val)

        # update height and rebalance every node that was traversed in recursive insertion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return root

    def min(self, root: Node) -> Node:
        """
        Return node with the smallest value in subtree rooted at `root`.

        :param root: root node of subtree to search for min.
        :return: node in the subtree rooted at `root` with the smallest value.
        """
        if root is None:
            return None
        if root.left is not None:
            return self.min(root.left)
        return root

    def max(self, root: Node) -> Node:
        """
        Return node with the largest value in subtree rooted at `root`.

        :param root: root node of subtree to search for max.
        :return: node in the subtree rooted at `root` with the largest value.
        """
        if root is None:
            return None
        if root.right is not None:
            return self.max(root.right)
        return root

    def search(self, root: Node, val: T) -> Node:
        """
        Search subtree rooted at `root` for a node containing `val`.
        If `val` is not found, return the node of which `val` would be a child.

        :param root: root node of subtree to search for `value`.
        :param val: value to search for in subtree rooted at `root`.
        :return: node in the subtree rooted at `root` with value `val`.
        """
        if root is None:
            return None
        elif val < root.value:
            return self.search(root.left, val) if root.left is not None else root
        elif val > root.value:
            return self.search(root.right, val) if root.right is not None else root
        return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder traversal of the subtree rooted at `root` using a generator.

        :param root: root node of subtree to traverse.
        :return: generator object which yields nodes through an inorder traversal
        """
        # inorder: left, center, right
        if root is not None:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a preorder traversal of the subtree rooted at `root` using a generator.

        :param root: root node of subtree to traverse.
        :return: generator object which yields nodes through a preorder traversal
        """
        # preorder: center, left, right
        if root is not None:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a postorder traversal of the subtree rooted at `root` using a generator.

        :param root: root node of subtree to traverse.
        :return: generator object which yields nodes through a postorder traversal
        """
        # postorder: left, right, center
        if root is not None:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a level-order (breadth-first) traversal of the subtree
        rooted at `root` using a generator.

        :param root: root node of subtree to traverse.
        :return: generator object which yields nodes through a level-order (breadth-first) traversal
        """
        # levelorder: implement breadth-first with queue
        if root is not None:
            q = queue.SimpleQueue()
            q.put(root)
            while not q.empty():
                current = q.get()
                yield current
                if current.left is not None:
                    q.put(current.left)
                if current.right is not None:
                    q.put(current.right)

    def remove(self, root: Node, val: T) -> Node:
        """
        Remove the node with `value` from the subtree rooted at `root` if it exists.
        Return the root node of the balanced subtree following removal.

        :param root: root node of subtree from which to remove.
        :param val: value to be removed from subtree.
        :return: root node of balanced subtree.
        """
        # handle empty and recursive left/right cases
        if root is None:
            return None
        elif val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # handle actual deletion step on this root
            if root.left is None:
                # pull up right child, set parent, decrease size, properly handle origin-reset
                if root is self.root:
                    self.root = root.right
                if root.right is not None:
                    root.right.parent = root.parent
                self.size -= 1
                return root.right
            elif root.right is None:
                # pull up left child, set parent, decrease size, properly handle origin-reset
                if root is self.root:
                    self.root = root.left
                if root.left is not None:
                    root.left.parent = root.parent
                self.size -= 1
                return root.left
            else:
                # two children: swap with predecessor and delete predecessor
                predecessor = self.max(root.left)
                root.value = predecessor.value
                root.left = self.remove(root.left, predecessor.value)

        # update height and rebalance every node that was traversed in recursive deletion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return root

    def __getitem__(self, key):
        found_node = self.search(self.root, key)
        if found_node is None:
            raise KeyError('Key doesn\'t exist')
        return found_node.value

    def __delitem__(self, key):
        self.remove(self.root, key)

    def __contains__(self, item):
        found_node = self.search(self.root, item)
        return found_node is not None and found_node.key == item

    def __setitem__(self, key, value):
        search_node = self.search(self.root, key)
        if search_node is None or search_node.key != key:
            self.insert(self.root, key, value)
        else:
            search_node.value = value