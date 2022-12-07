# DO NOT MODIFY
# My implementation of red-black tree from CSE 331
from __future__ import annotations
from typing import TypeVar, Generic, Tuple, Generator
import queue
from copy import deepcopy

T = TypeVar('T')


class Node:
    """
    A red/black tree node class
    :value: stored value, also serves as key. Generic Type.
    :is_red: boolean identifier
    :parent: reference to parent node
    :left: reference to left node
    :right: reference to right node
    """

    __slots__ = ['key', 'is_red', 'parent', 'left', 'right', 'value']

    def __init__(self, key: Generic[T], is_red: bool = True, parent: Node = None,
                 left: Node = None, right: Node = None, value: Generic[T] = None) -> None:
        """ Node Initializer"""
        self.key = key
        self.value = value
        self.is_red = is_red
        self.parent = parent
        self.left, self.right = left, right

    def __eq__(self, other: Node) -> bool:
        """ Node Equality Comparator """
        return (type(None if not self else self.key) is type(None if not other else other.key) and
                self.key == other.key and
                self.is_red == other.is_red)

    def __str__(self) -> str:
        """ returns string representation of a node """
        r = '(R)' if self.is_red else '(B)'
        return str(self.key) + ' ' + r

    def __repr__(self) -> str:
        return self.__str__()

    def subtree_size(self) -> int:
        """ returns size of tree rooted at given node """
        return 1 + (self.left.subtree_size() if self.left else 0) + (self.right.subtree_size() if self.right else 0)

    def subtree_height(self) -> int:
        """ returns the height of a subtree rooted at a given node """
        return 1 + max(self.left.subtree_height() if self.left else 0, self.right.subtree_height() if self.right else 0)

    def subtree_redblack_property(self) -> bool:
        """ returns whether a tree adheres to red black properties """

        def rb_check_helper(node) -> Tuple[bool, int]:
            """ recursive helper """
            if not node:
                return True, 1
            if not node.parent and node.is_red:
                return False, 0
            if node.is_red and ((node.left and node.left.is_red) or (node.right and node.right.is_red)):
                return False, -1

            left_check, num_black_left = rb_check_helper(node.left)
            right_check, num_black_right = rb_check_helper(node.right)
            return all([left_check, right_check, num_black_left == num_black_right]), num_black_left + (0 if node.is_red else 1)

        return rb_check_helper(self)[0]



class RBtree:
    """
    A Red/Black Tree class
    :root: Root Node of the tree
    :size: Number of Nodes
    """

    __slots__ = ['root', 'size']

    def __init__(self, root: Node = None):
        """ Initializer for an RBtree """
        # this alllows us to initialize by copying an existing tree
        self.root = deepcopy(root)
        if self.root:
            self.root.parent = None
        self.size = 0 if not self.root else self.root.subtree_size()

    def __eq__(self, other: RBtree) -> bool:
        """ Equality Comparator for RBtrees """
        comp = lambda n1, n2: n1 == n2 and (
            (comp(n1.left, n2.left) and comp(n1.right, n2.right)) if (n1 and n2) else True)
        return comp(self.root, other.root) and self.size == other.size

    def __str__(self) -> str:
        """ represents Red/Black tree as string """

        if not self.root:
            return 'Empty RB Tree'

        root, bfs_queue, height = self.root, queue.SimpleQueue(), self.root.subtree_height()
        track = {i: [] for i in range(height + 1)}
        bfs_queue.put((root, 0, root.parent))

        while bfs_queue:
            n = bfs_queue.get()
            if n[1] > height:
                break
            track[n[1]].append(n)
            if n[0] is None:
                bfs_queue.put((None, n[1] + 1, None))
                bfs_queue.put((None, n[1] + 1, None))
                continue
            bfs_queue.put((None, n[1] + 1, None) if not n[0].left else (n[0].left, n[1] + 1, n[0]))
            bfs_queue.put((None, n[1] + 1, None) if not n[0].right else (n[0].right, n[1] + 1, n[0]))

        spaces = 12 * (2 ** (height))
        ans = '\n' + '\t\tVisual Level Order Traversal of RBtree'.center(spaces) + '\n\n'
        for i in range(height):
            ans += f"Level {i + 1}: "
            for n in track[i]:
                space = int(round(spaces / (2 ** i)))
                if not n[0]:
                    ans += ' ' * space
                    continue
                ans += "{} ({})".format(n[0], n[2].key if n[2] else None).center(space, " ")
            ans += '\n'
        return ans

    def __repr__(self) -> str:
        return self.__str__()

    ################################################################
    ################### Complete Functions Below ###################
    ################################################################

    ######################## Static Methods ########################
    # These methods are static as they operate only on nodes, without explicitly referencing an RBtree instance

    @staticmethod
    def set_child(parent: Node, child: Node, is_left: bool) -> None:
        """
        Sets the child parameter of parent to child.
        Which child is determined by the identifier is_left.
        The parent parameter of the new child node should be updated as required.
        :param parent: parent of child
        :param child: child of parent
        :param is_left: Determine which side to put child
        :return: None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if is_left:
            parent.left = child
        else:
            parent.right = child
        if child:
            child.parent = parent

    @staticmethod
    def replace_child(parent: Node, current_child: Node, new_child: Node) -> None:
        """
        Replaces parent's child current_child with new_child.
        :param parent: parent of child
        :param current_child: current child
        :param new_child: new child
        :return: None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if parent.left == current_child:
            RBtree.set_child(parent, new_child, True)
        else:
            RBtree.set_child(parent, new_child, False)

    @staticmethod
    def get_sibling(node: Node) -> Node:
        """
        Given a node, returns the other child of that node's parent,
        or None should no parent exist.
        :param node: Node to find sibling
        :return RBNode : sibling of Node or None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.parent is None:
            return None
        if node.parent.left == node:
            return node.parent.right
        return node.parent.left

    @staticmethod
    def get_grandparent(node: Node) -> Node:
        """
        Given a node, returns the parent of that node's parent,
        or None should no such node exist.
        :param node: node to find grandparent
        :return: RBNode : Grandparent of node or None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.parent is None:
            return None
        return node.parent.parent

    @staticmethod
    def get_uncle(node: Node) -> Node:
        """
        Given a node, returns the sibling of that node's parent,
         or None should no such node exist.
        :param node: Node to find uncle
        :return: RBNode : uncle of node or None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.parent is None:
            return None
        return RBtree.get_sibling(node.parent)

    ######################## Misc Utilities ##########################

    def min(self, node: Node) -> Node:
        """
        Returns the minimum value stored in
        the subtree rooted at node. (None if the subtree is empty).
        :param node: root node of subtree
        :return: min Node or None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """

        def find_min(current: Node):
            if current.left is None:
                return current
            return find_min(current.left)

        if node is None:
            return None
        return find_min(node)

    def max(self, node: Node) -> Node:
        """
        Returns the maximum value stored in
        the subtree rooted at node. (None if the subtree is empty).
        :param node: root node of subtree
        :return: max Node or None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """

        def find_max(current: Node):
            if current.right is None:
                return current
            return find_max(current.right)

        if node is None:
            return None
        return find_max(node)

    def search(self, node: Node, val: Generic[T]) -> Node:
        """
        Searches the subtree rooted at node for a node containing value val.
        If such a node exists, return that node- otherwise
        return the node which would be parent
        to a node with value val should such a node be inserted.
        :param node: root node of subtree
        :param val: value to find
        :return: node exists of value or possible parent
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """

        def find_node(current: Node):
            if current.key == val:
                return current

            if val < current.key:
                if current.left is None:
                    return current
                return find_node(current.left)

            if current.right is None:
                return current
            return find_node(current.right)

        if node is None:
            return None
        return find_node(node)

    ######################## Tree Traversals #########################

    def inorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing an inorder
        traversal of the subtree rooted at node.
        :param node: root of subtree
        :yield: inorder of subtree
        Time Complexity: O(n), Space Complexity: O(n)
        """

        def inorder_travel(current: Node):
            """
            Recusion function support to find inorder of subtree
            :param current: current node
            :yield inorder order
            """
            if current is None:
                return
            yield from inorder_travel(current.left)
            yield current
            yield from inorder_travel(current.right)

        yield from inorder_travel(node)

    def preorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing an preorder
        traversal of the subtree rooted at node.
        :param node: root of subtree
        :yield: preorder of subtree
        Time Complexity: O(n), Space Complexity: O(n)
        """

        def preorder_travel(current: Node):
            """
            Recusion function support to find preorder of subtree
            :param current: current node
            :yield preorder order
            """
            if current is None:
                return
            yield current
            yield from preorder_travel(current.left)
            yield from preorder_travel(current.right)

        yield from preorder_travel(node)

    def postorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing an postorder
        traversal of the subtree rooted at node.
        :param node: root of subtree
        :yield: postorder of subtree
        Time Complexity: O(n), Space Complexity: O(n)
        """

        def postorder_travel(current: Node):
            """
            Recusion function support to find preorder of subtree
            :param current: current node
            :yield preorder order
            """
            if current is None:
                return
            yield from postorder_travel(current.left)
            yield from postorder_travel(current.right)
            yield current

        yield from postorder_travel(node)

    def bfs(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing an breath first search
        traversal of the subtree rooted at node.
        :param node: root of subtree
        :yield: Generator
        Time Complexity: O(n), Space Complexity: O(n)
        """
        bfs_queue = queue.Queue()
        bfs_queue.put(node)
        while not bfs_queue.empty():
            current = bfs_queue.get()
            if current is None:
                continue
            yield current
            bfs_queue.put(current.left)
            bfs_queue.put(current.right)

    ################### Rebalancing Utilities ######################

    def left_rotate(self, node: Node) -> None:
        """
        Performs a left tree rotation on the subtree rooted at node.
        :param node: node to perform left rotation
        :return:  None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        # Changing parent
        if node.parent is not None:
            RBtree.replace_child(node.parent, node, node.right)
        else:
            self.root = node.right
            self.root.parent = None

        right_left_node = node.right.left
        RBtree.set_child(node.right, node, is_left=True)
        RBtree.set_child(node, right_left_node, is_left=False)

    def right_rotate(self, node: Node) -> None:
        """
        Performs a right tree rotation on the subtree rooted at node.
        :param node: node to perform right rotation
        :return:  None
        Time Complexity: O(1), Space Complexity: O(1)
        """
        # Changing parent
        if node.parent is not None:
            RBtree.replace_child(node.parent, node, node.left)
        else:
            self.root = node.left
            self.root.parent = None

        left_right_node = node.left.right
        RBtree.set_child(node.left, node, is_left=False)
        RBtree.set_child(node, left_right_node, is_left=True)

    def insertion_repair(self, node: Node) -> None:
        """
        This method is not tested explicitly,
        but should be called after insertion on the node which was inserted,
        and should rebalance the tree by ensuring adherance to Red/Black properties.
        It is highly recommended you utilize recursion.
        :param node: node to check rebalanced tree
        :return: None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """
        grandparent = RBtree.get_grandparent(node)
        parent = node.parent
        uncle = RBtree.get_uncle(node)
        # If node is root, node's color is black
        if parent is None:
            node.is_red = False
            return
        # If node is black, return
        if not parent.is_red:
            return
        # Case of parent and uncle are red
        if parent and uncle and parent.is_red and uncle.is_red:
            # Changing uncle and parent to black
            parent.is_red = False
            uncle.is_red = False
            # Grandparent to red
            grandparent.is_red = True
            # fixing balance on grandparent
            self.insertion_repair(grandparent)
            return
        # If node is parent'right, and parent is grandparent's left
        # Perform left rotation at parent, and swap parent, and node
        if node == parent.right and parent == grandparent.left:
            self.left_rotate(parent)
            parent, node = node, parent

        # If node is parent'left, and parent is grandparent's right
        # Perform right rotation at parent, and swap parent, and node
        if node == parent.left and parent == grandparent.right:
            self.right_rotate(parent)
            parent, node = node, parent

        # parent to black, grandparent to red
        parent.is_red = False
        grandparent.is_red = True

        # rotation for balance of three nodes
        if parent.left == node:
            self.right_rotate(grandparent)
        else:
            self.left_rotate(grandparent)

    @staticmethod
    def is_not_none_or_red(node: Node) -> bool:
        """
        Checking node is exist or red node
        :param node: node to check
        :return: boolean
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node is None:
            return False
        return node.is_red

    @staticmethod
    def is_none_or_black(node: Node) -> bool:
        """
        Checking node is none or black node
        :param node: node to check
        :return: boolean
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node is None:
            return True
        return not node.is_red

    @staticmethod
    def both_children_black(node: Node) -> bool:
        """
        Checking if both children are black
        :param node: node to check
        :return: both children need to be black for true
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.left and node.left.is_red:
            return False
        if node.right and node.right.is_red:
            return False
        return True

    def __remove_case1(self, node: Node) -> bool:
        """
        Removing Case1 : node is red or parent is None
        :param node: node to check case
        :return: checking status
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.is_red or node.parent is None:
            return True
        return False

    def __remove_case2(self, node: Node, sibling: Node) -> bool:
        """
        Removing Case2 : sibling is red.
        If status is true,
        then, parent and sibling to be black
        if node is parent's left, perform left rotation at parent
        otherwise, right rotation
        :param node: node to check
        :param sibling: sibling of node
        :return: status of condition
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if not sibling.is_red:
            return False

        node.parent.is_red = True
        sibling.is_red = False
        if node == node.parent.left:
            self.left_rotate(node.parent)
        else:
            self.right_rotate(node.parent)
        return True

    def __remove_case3(self, node: Node, sibling: Node) -> bool:
        """
        Remove Case3: parent black, both sibling's children black
        if true, sibling to red, recursively call prepare remove(parent)
        :param node: node to check
        :param sibling: sibling of node
        :return: status of condition
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """
        if (not node.parent.is_red) and RBtree.both_children_black(sibling):
            sibling.is_red = True
            self.prepare_removal(node.parent)
            return True
        return False

    def __remove_case4(self, node: Node, sibling: Node) -> bool:
        """
        Remove Case4: parent red, both sibling's children black
        if true, parent to black, sibling to red
        :param node: node to check
        :param sibling: sibling of node
        :return: status of condition
        Time Complexity: O(1), Space Complexity: O(1)
        """
        if node.parent.is_red and RBtree.both_children_black(sibling):
            sibling.is_red = True
            node.parent.is_red = False
            return True
        return False

    def __remove_case5(self, node: Node, sibling: Node) -> bool:
        """
        Remove Case 5: condition is
        sibling's left is red, right is black, node is left of parent
        sibling to red, and left of sibling to black
        perform right rotation at sibling
        :param node: node to check
        :param sibling: sibling of node
        :return: status of condition
        """
        if RBtree.is_not_none_or_red(sibling.left):
            if RBtree.is_none_or_black(sibling.right):
                if node == node.parent.left:
                    sibling.is_red = True
                    sibling.left.is_red = False
                    self.right_rotate(sibling)
                    return True
        return False

    def __remove_case6(self, node: Node, sibling: Node) -> bool:
        """
        Remove Case 6: condition is
        sibling's left is black, right is red, node is right of parent
        sibling to red, and right of sibling to black
        perform left rotation at sibling
        :param node: node to check
        :param sibling: sibling of node
        :return: status of condition
        """
        if RBtree.is_none_or_black(sibling.left):
            if RBtree.is_not_none_or_red(sibling.right):
                if node == node.parent.right:
                    sibling.is_red = True
                    sibling.right.is_red = False
                    self.left_rotate(sibling)
                    return True
        return False

    def prepare_removal(self, node: Node) -> None:
        """
        This method is not tested explicitly,
        but should be called prior to removal,
        on a node that is to be removed.
        It should ensure balance is maintained after the removal.
        :param node: node to ensure the balance
        :return: None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """

        if self.__remove_case1(node):
            return

        sibling = self.get_sibling(node)
        if self.__remove_case2(node, sibling):
            sibling = self.get_sibling(node)

        if self.__remove_case3(node, sibling):
            return

        if self.__remove_case4(node, sibling):
            return

        if self.__remove_case5(node, sibling):
            sibling = self.get_sibling(node)

        if self.__remove_case6(node, sibling):
            sibling = self.get_sibling(node)

        sibling.is_red = node.parent.is_red
        node.parent.is_red = False
        if node == node.parent.left:
            sibling.right.is_red = False
            self.left_rotate(node.parent)
        else:
            sibling.left.is_red = False
            self.right_rotate(node.parent)

    ##################### Insertion and Removal #########################

    def insert(self, node: Node, val: Generic[T], real_val: Generic[T] = None) -> None:
        """
        Inserts an RBnode object to the subtree rooted at node with value val.
        Should a node with value val already exist in the tree, do nothing.
0        :param node: root of subtree
        :param val: value to insert
        :param real_val: value of the node
        :return: None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """
        insert_node = Node(val, True, value=real_val)

        if self.root is None:
            self.root = insert_node
            insert_node.parent = None
            self.insertion_repair(insert_node)
            self.size += 1
            return
        # Finding the insert point
        insert_point = self.search(node, val)
        if insert_point.key == val:
            return

        self.size += 1
        RBtree.set_child(insert_point, insert_node, insert_node.key < insert_point.key)
        self.insertion_repair(insert_node)

    def remove(self, node: Node, val: Generic[T]) -> None:
        """
        Removes node with value val from the subtree rooted at node. If no such node exists, do nothing.
        :param node: root of subtree
        :param val: value to remove
        :return: None
        Time Complexity: O(log(n)), Space Complexity: O(1)
        """
        # Finding remove node
        removed_node = self.search(node, val)
        if removed_node is None:
            return
        if removed_node.left is not None and removed_node.right is not None:
            next_value_node = self.max(removed_node.left)
            next_value = next_value_node.key
            self.remove(next_value_node, next_value)
            removed_node.key = next_value
            return

        if not removed_node.is_red:
            self.prepare_removal(removed_node)

        # BST removing node from either having 0 child, or 2 children
        substitute_node = removed_node.left if removed_node.left else removed_node.right

        if removed_node.parent is None:
            self.root = substitute_node
            if substitute_node:
                substitute_node.parent = None
                substitute_node.is_red = False
        elif removed_node.parent.left == removed_node:
            RBtree.set_child(removed_node.parent, substitute_node, True)
        else:
            RBtree.set_child(removed_node.parent, substitute_node, False)

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

