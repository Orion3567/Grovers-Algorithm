import numpy as np
import qpython as qp
import random

__author__ = 'Aidan Shafer'
__date__ = '5/31/2015'

"""
Applcation of Grovers Algorithm through Simulation of a Quantum System.

Given an ordered list and an item, this algorithm is capable of finding the
index of the item. If the item appears more than once, the algorithm has a
tendency to 'overshoot', although it still often finds one instance. While
horrendously inefficient on a classical machine, it would be maximally efficient
on a quantum machine.

A quantum computer is simulated with the state class. It is capable of
simulating the application of all gates needed for Grovers Algorithm and more.
I plan on continuing to add to it in the future.
"""

#############################  STATE CLASS  ####################################

class state:
    """
    The quantum state. Essentially a simulated quantum computer, capable of many
    basic quantum operations.
    """
    def __init__(self, bits=[[1, 0]]):
        self.size = len(bits)

        self.vec = np.array(bits[-1])
        for bit in bits[:-1][::-1]:
            self.vec = np.concatenate([bit[0]*self.vec, bit[1]*self.vec])

    def normalize(self):
        norm = np.sqrt( np.sum( np.abs(self.vec)**2 ) )
        self.vec /= norm

    def add_bit(self, bit=[1,0], pos=0):
        if pos == -1:
            pos = self.size

        m = 2**pos
        n = 2**(self.size-pos)

        newstate = np.array([])
        for stack in range(m):
            newstate = np.concatenate( [ newstate, bit[0]*self.vec[stack*n:(stack+1)*n],
                                            bit[1]*self.vec[stack*n:(stack+1)*n]] , axis=None )

        self.vec = newstate

    def measure(self, bit):
        self.normalize() # Just in case a normalization constant is dropped

        # See what the state would be after measurement
        stateif0 = []
        stateif1 = []

        chunkct = 0
        chunk = 2**(self.size-bit)
        partof0state = True

        for bitstate in self.vec:
            if partof0state:
                stateif0.append(bitstate)
            else:
                stateif1.append(bitstate)

            chunkct += 1
            if chunkct == chunk:
                chunkct = 0
                partof0state = not partof0state

        # Check the probability and 'measure' with a random number.
        p0 = np.sum( [np.abs(prob)**2 for prob in stateif0] )
        if random.random() < p0:
            measured_state = 0
            self.vec = np.array(stateif0)
        else:
            measured_state = 1
            self.vec = np.array(stateif1)

        # Renormalize and update size
        self.normalize()
        self.size -= 1

        # Return value if desired.
        return measured_state

    def _singlequbitgate(self, bits, mtrx):
        """
        Apply single qubit gates on certain qubits.
        """
        opts = []
        for ct in range(1, self.size+1):
            if ct in bits:
                opts.append( mtrx )
            else:
                opts.append( Identity_Matrix )

        operator = tprod(*opts)

        self.vec = operator @ self.vec

    def Hadamard(self, bits=[]):
        self._singlequbitgate(bits, Hadamard_Matrix)

    def SigX(self, bits=[]):
        self._singlequbitgate(bits, Sigx_Matrix)

    def SigY(self, bits=[]):
        self._singlequbitgate(bits, Sigy_Matrix)

    def SigZ(self, bits=[]):
        self._singlequbitgate(bits, Sigz_Matrix)


    def _controlgate(self, control, target, mtrx, swapbit=False):
        """
        A controlled-U gate. Accepts any number of controls but only one target.
        """
        # Force control into a list
        if isinstance(control, int):
            control = list(control)
        lencontrol = len(control)

        # Organize bits to simplify "circuitry"
        self.SWAP(target, self.size)
        if self.size in control:
            control[np.where(control == self.size)[0]] = target

        # Link a junction if it's a control or just pass
        operator = mtrx
        for ct in range(1, self.size)[::-1]:
            if ct in control: #Order shouldn't matter
                operator = np.concatenate([np.concatenate( [Iden_mtrx(operator), np.zeros_like(operator)], axis=1 ),
                                            np.concatenate( [np.zeros_like(operator), operator], axis=1  )], axis=0 )
            else:
                operator = tprod(Identity_Matrix, operator )

        # Apply
        self.vec = operator@self.vec

        # Switch bit back
        self.SWAP(target, self.size)


    def CNOT(self, control, target):
        self._controlgate(control, target, Sigx_Matrix, swapbits)

    def cY(self, control, target):
        self._controlgate(control, target, Sigy_Matrix, swapbits)

    def cZ(self, control, target):
        self._controlgate(control, target, Sigz_Matrix)

    def SWAP(self, bita, bitb):
        """
        Qubit Swap.
        """
        if bita == bitb:
            return
        matsize = 2**self.size
        abit = self.size-bita
        bbit = self.size-bitb

        indices = swap( np.arange(matsize), abit, bbit)

        operator = np.zeros((matsize, matsize)) # Not an Identity!
        for ct in range(matsize):
            operator[ct, indices[ct]] = 1

        self.vec = operator@self.vec

    def custom_gate(self, mtrx):
        """
        Apply your own matrix to the state vector.
        """
        self.vec = mtrx@self.vec

    def custom_1bitgate(self, mtrx, bitn=0):
        """
        Apply your own matrix to a qubit.
        """
        opts = []
        for ct in range(self.size):
            if ct == bitn:
                opts.append( mtrx )
            else:
                opts.append( Identity_Matrix )

        operator = tprod(*opts)

        self.vec = operator @ self.vec

    def __str__(self):
        statestr = str()
        for ct in range(len(self.vec)):
            if self.vec[ct]:
                statestr += '{2:0.2f}*|{1:0{0:d}b}>\n'.format(self.size, ct, self.vec[ct])
        return statestr

##########################  GROVERS ALGORITHM  #################################

def GroversAlgorithm(database, item, update=False):
    """
    Grovers Algorithm - A quantum algorithm for finding an item in an ordered list.

    'Database' is the ordered list and 'item' is the object being searched for.
    This algorithm is able to find the item with high probability (~94% for a
    list of length 5-8).

    The only 'mysterious' quantum gate used is the Oracle operator. Every other
    computation is done using typical quantum gates. Namely, the other
    'mysterious' operator, X (chi), is expressed solely in terms of Pauli-Z spin
    matrices and it's controlled versions.

    Since this algorithm only simulates a quantum machine, it is horribly
    inefficient. In fact, the item is 'found' by getting the oracle operator in
    the very beginning. However, on a true quantum device this algorithm is much
    more efficient than any classical algorithm (it's actually maximally
    efficient).
    """
    def _ChiOp(system, nbits):
        """
        The X (Chi) Operator.
        Essentially an identity, but negative everywhere but |00..0>
        """
        # Apply sigZ gate to all bits
        system.SigZ( range(1, nbits+1) )

        # A cZ between _ALL_ bits (presumably this is just one 'operation' as it
        # would be one long junction, if not the N operations make it very
        # computationally taxing)
        for control_target in combocounter(nbits):
            system.cZ(control_target[:-1], control_target[-1])

    # Defines the oracle operator. I will leave it to the quantum realm and just
    # use the matrix.
    Omtrx, size = Oracle(database, item)

    nbits = round( np.log(size) / np.log(2) ) # size(N) = 2**nbits
    allbits = range(1, nbits+1) # A list of all bits in the system (i.e. 1st 2nd ...)

    # This is the quantum computer that's being simulated.
    system = state( str_to_bits('0'*nbits) )


    ##--   ALGORITHM   --##

    # Apply a Hadamard to equally occupy all states.
    system.Hadamard( allbits )

    for ct in range(int( np.sqrt(size) - 1 )):
        # An optional state update. In real life this would not be possible without ruining the calculation.
        if update:
            print( '{}\n\n'.format(system) )

        # The 'only' four operations that need to be done.
        system.custom_gate(Omtrx)
        system.Hadamard(allbits)
        _ChiOp(system, nbits)
        system.Hadamard(allbits)

    # Final update.
    if update:
        print( '{}\n\n'.format(system) )

    # Measurement of the system, one qubit at a time.
    measured_state = []
    for bit in range(nbits):
        measured_state.append( system.measure(1) ) #The first bit changes with each measure.

    return measured_state


def Oracle(database, item):
    """
    Given an ordered (but not necessarily sorted) database. This will perform
    the 'oracle' operation from Grovers Algorithm. In truth, this classical
    operation finds our object without the need of futher calculation. However,
    in the quantum realm this operation is much quicker for large databases, as
    well as the linear algebra to follow.
    """
    datalen = len(database)
    size = 2 ** ( int( np.log(datalen) / np.log(2) ) + 1 )  # So its of form 2^n - 1

    omtrx = Iden_mtrx(size)

    # The cheat
    index = np.where(np.array(database) == item)

    omtrx[index, index] = -1

    return omtrx, size

#########################  AUXILLARY FUNCTIONS  ################################

def tprod(*args):
    """
    2-D Tensor product of all arguments.

    i.e.            |0 0 1 0|
    |0 1|   |1 0|   |0 0 0 1|
    |1 0| X |0 1| = |1 0 0 0|
                    |0 1 0 0|
    """
    ans = args[-1]
    for mat in args[:-1][::-1]:

        newmat = []
        for arow in mat:
            newrow = []

            for aitm in arow:
                newrow.append(aitm*ans)

            newmat.append( np.concatenate(newrow, axis=1) )
        ans = np.concatenate(newmat, axis=0)

    return ans

def combocounter(lstsize):
    """
    Returns all combinations of numbers counter.

    i.e. for 1,2,3:
    1, 2, 3, 12, 13, 23, 123
    """
    bitlist = []
    for n_ctrl in range(1, lstsize):
        bits = np.arange(n_ctrl+1)+1 # Range from 1 to n_ctrl

        unfinished = True
        while unfinished:
            bitlist.append( np.array(bits) )

            bits[-1] += 1
            if bits[-1] == lstsize+1:
                bits[-1] -= 1
                searching = True
                ctrl_to_move = -2
                while searching:
                    if -ctrl_to_move > n_ctrl+1:
                        unfinished = False
                        searching = False
                    elif bits[ctrl_to_move]+1 == bits[ctrl_to_move+1]:
                        ctrl_to_move -= 1
                    else:
                        searching = False
                        bits[ctrl_to_move] += 1
                        bits[ctrl_to_move+1:] = range(bits[ctrl_to_move]+1, bits[ctrl_to_move]-ctrl_to_move)
    return bitlist

def str_to_bits(bitstr):
    """
    For use with state class. Allows user to enter '000' for three bits
    initialized to zero [[1,0],[1,0],[1,0]].
    """
    statelst = []

    for bit in bitstr:
        if bit == '0':
            newvec = [1, 0]
        elif bit == '1':
            newvec = [0, 1]

        statelst.append(newvec)

    return statelst

def binlst2index(binlst):
    """
    Converts a list of binary digits to a number.
    """
    index = 0
    binlen = len(binlst)

    for ct in range( binlen ):
        index += binlst[binlen-ct-1] * 2**ct

    return index


def swap(x, bita, bitb):
    """
    Swaps bits in a list of numbers, with bit number counted from the left.
    i.e. given x = 11 = 1011(bin), and bit placements 1 and 2, this returns
    7 = 0111(bin)
    """
    set1 =  (x >> bita) & 1
    set2 =  (x >> bitb) & 1

    xor = (set1 ^ set2)
    xor = (xor << bita) | (xor << bitb)
    return x ^ xor

def Iden_mtrx(size):
    """
    Identity Matrix of any size.
    """
    # Size determination based on whether on it's an integer or a matrix
    if isinstance(size, int):
        mtrx = np.zeros((size, size))
    if isinstance(size, np.ndarray):
        mtrx = np.zeros_like(size)
        size = mtrx.shape[0]
    for ct in range(size):
        mtrx[ct, ct] = 1
    return mtrx



################################################################################

Identity_Matrix = np.array( [[1, 0], [0, 1]] )

Hadamard_Matrix = np.array( [[1, 1], [1, -1]] ) * np.sqrt(2)/2

Sigx_Matrix = np.array( [[0, 1], [1, 0]] )

Sigy_Matrix = np.array( [[0, -1j], [1j, 0]] )

Sigz_Matrix = np.array( [[1, 0], [0, -1]] )

CNOT_Matrix = np.array( [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]] )

################################################################################

database = list( 'sphinx of black quartz, judge my vow' )
#database = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
query = 'q'

#database = range(100)
#query = 42



binval = GroversAlgorithm(database, query, update=True)
index = binlst2index(binval)



if index > len(database):
    result = "Failure"
    itemfound = 'NonValid'
elif database[index] == query:
    result = "Success!"
    itemfound = database[index]
else:
    result = "Failure"
    itemfound = database[index]

print('Result: {}\n\t\t-->{}\n\nItem: {}\n{}'.format(  binval, index, itemfound, result ))
