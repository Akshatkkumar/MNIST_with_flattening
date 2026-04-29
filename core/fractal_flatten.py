import numpy as np 
import matplotlib.pyplot as plt 
import ast

# cyclic group Z_4 also fourth roots of unity represent orientation group for flattening 
# u = 1, l = i, d = -1, r = -i
# consider orientation as a change in the connecting curve, not the block
# fill only connections available on each subdivision(connection points)

# flatten a 2d array normally, uses a Z curve
def flatten(l):
    return np.asarray(l).flatten()

# Flatten a 2d with an S curve
def flatten_s(l):
    fl = []
    c = 0
    for kl in l:
        kl = list(kl)
        if c % 2 == 1:
            kl.reverse()
        fl = fl + kl
        c += 1
    return fl

# flatten a (square) array with a space filling Hilbert curve(currently just expands out to a 2^n x 2^n curve and conserves the actual dimensions present)
# Note to self, rework this into something actually generalized to nxn images
def flatten_adv(l):
    # Converts a 1D index 'd' to 2D coordinates (x, y) on a Hilbert curve of size n x n.
    def d2xy(n, d):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    # Helper for flipping quadrants
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            return y, x
        return x, y

    n = int(2**(np.ceil(np.log(len(l))/np.log(2))))  # Smallest power of 2 for 28x28
    mapping = [d2xy(n, i) for i in range(n*n)]

    # Filter mapping to only include points within the 28x28 bounds
    true_mapping = [(x, y) for x, y in mapping if x < len(l) and y < len(l)]
    return  [l[y][x] for x,y in true_mapping]

# Flatten using a given complete mapping/curve
# Flattens a given square 2d array with a given mapping dict 
def flatten_gen(l2d, mapp):
    l = [None] * len(mapp.keys())
    for i in range(len(mapp.keys())):
        coord = stol(mapp[i])
        l[i] = l2d[coord[0]][coord[1]]
    return(l)

# Taxicab distance
def Txd(p,q):
    return np.sum(np.abs(np.subtract(p,q)))

# Euclidean distance
def Ecd(p,q):
    return np.sqrt(np.dot((v:=np.subtract(p,q)),v))

# H is a dictionary that maps a single 1D array to a 2D array -> array has to be in a string to be hashable
# I is the list of index's of the original space
# T is the distance function(usually either euclidean or Taxicab)
# Eg, I = [[0,0],[0,1],[1,0],[1,1]], H = {1: '[0,0]', 2: '[0,1]', 3:'[1,0]', 4:'[1,1]'} (This is a z shape curve), T = Txd
# Score gives the distance preserving score of the 
def score(H, T=Txd, power=1):
    H_inv = {H[k]:k for k in H.keys()}
    I = [stol(H[k]) for k in H.keys()]
    scores = []
    c = [(max([x[0] for x in I]) - min([x[0] for x in I]))/2, (max([x[1] for x in I]) - min([x[1] for x in I]))/2]
    for p in I:
        for q in I:
            if p != q:
                scores.append(np.multiply(T(c,q),np.multiply(T(c,p),np.divide(np.pow(np.abs(T(p,q) - T(H_inv[ltos(p)],H_inv[ltos(q)])),power),T(p,q)))))
    return scores

# Scores the varience from point to point on a given dataset given a mapping
def score_dataset(mapp, X):
    np.mean([sum(np.abs(np.diff(flatten_gen(X[i],mapp)))) for i in range(500)])

# turns a stringed list of homogenus type eg. '[0,1,2]' into the original list [0,1,2]
def stol(s,typ=int):                         
    l = []
    cn = ''
    for i in s[1:]:
        if i == ']' or i == ',':
            l.append(typ(cn))
            cn = ''
        else:
            cn = cn + i
    return l

# convert list to a string
def ltos(l):
    return str(l).replace(' ','')

# plots a given mapping in format as a curve and prints numbers for mapping value
# nums_visible decides if numbers are visible in the image
def plotmapping(mapping, nums_visible=True):
    coords = [ast.literal_eval(mapping[i]) for i in range(len(mapping))]

    y_coords = [c[0] for c in coords]
    x_coords = [c[1] for c in coords]

    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker='o', color='b', linestyle='-', linewidth=2)

    if nums_visible:
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.text(x + 0.1, y + 0.1, str(i), fontsize=12)

    plt.gca().invert_yaxis() # Invert to match array indexing (0,0 at top-left)
    plt.grid(True)
    plt.title("Path Visualization (2D Array Indices)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

# 4x4 Hilbert curve
# {0: '[0,0]', 1: '[0,1]', 2: '[1,1]', 3: '[1,0]', 4: '[2,0]', 5: '[3,0]', 6: '[3,1]', 7: '[2,1]', 8: '[2,2]', 9: '[3,2]', 10:'[3,3]', 11: '[2,3]', 12: '[1,3]', 13: '[1,2]', 14: '[0,2]', 15: '[0,3]'}, fl.Txd
# 4x4 Z curve 
# {0: '[0,0]', 1: '[0,1]', 2: '[0,2]', 3: '[0,3]', 4: '[1,0]', 5: '[1,1]', 6: '[1,2]', 7: '[1,3]', 8: '[2,0]', 9: '[2,1]', 10:'[2,2]', 11: '[2,3]', 12: '[3,0]', 13: '[3,1]', 14: '[3,2]', 15: '[3,3]'}, fl.Txd
# 4x4 S curve 
# {0: '[0,0]', 1: '[0,1]', 2: '[0,2]', 3: '[0,3]', 4: '[1,3]', 5: '[1,2]', 6: '[1,1]', 7: '[1,0]', 8: '[2,0]', 9: '[2,1]', 10:'[2,2]', 11: '[2,3]', 12: '[3,3]', 13: '[3,2]', 14: '[3,1]', 15: '[3,0]'}, fl.Txd

### Random curve mapping ###
def rand(n):
    mapp = {}
    l = list(range(n**2))
    for i in range(n):
        for j in range(n):
            v = np.random.choice(l)
            l.pop(l.index(v))
            mapp[int(v)] = ltos([i,j])
    myKeys = list(mapp.keys())
    myKeys.sort()

    # Sorted Dictionary
    sd = {i: mapp[i] for i in myKeys}
    return sd
    

### Z curve ### 
# makes a z curve mapping for a square grid of size n
def Z(n):
    mapp = {}
    curri = 0
    for i in range(n):
        for j in range(n):
            mapp[curri] = f'[{i},{j}]'
            curri += 1
    return(mapp)


### S curve ### 
def S(n):
    mapp = {}
    curri = 0
    for i in range(n):
        for j in range(n):
            if i % 2 == 0:
                mapp[curri] = f'[{i},{j}]'
            else:
                mapp[curri] = f'[{i},{n-j-1}]'
            curri += 1
    return(mapp)

### Hilbert-L curve algorithm using roots of unity ###
## Classes 
# class to contain a given qHilbert block(non-L)
class qhil_block:
    # Constructor method (initializes instance variables)
    # defines the bounds of the block(pair of list indices) and the orientation('u','l','r','d')
    def __init__(self, tli, tri, bli, bri, orientation):
        # index of top right, top left, bottom right, bottom left
        self.tri = tri  
        self.tli = tli    
        self.bri = bri
        self.bli = bli 
        self.blocksize = 1 + abs(tri[1] - tli[1])
        self.orientation = orientation

    # Checks if a given block is of size 1
    def check_one(self):
        return self.tri==self.tli==self.bri==self.bli

    # takes a given pair and makes a step
    # eg given 
    def step(self, p, s):
        if s == 'u':
            return (p[0]-1, p[1])
        elif s == 'l':
            return (p[0], p[1]-1)
        elif s == 'd':
            return (p[0]+1,p[1])
        elif s == 'r':
            return (p[0], p[1]+1)

    # Split into quadrants, or make an L, returns the new children objects or object + L
    def split(self):
        # every split creates connections, this will store them and return it at the end
        new_connects = []
        # L split 
        if self.blocksize % 2 == 1:
            if self.orientation == 'u':
                child_block = qhil_block(self.tli, self.step(self.tri, 'l'), self.step(self.bli, 'u'), self.step(self.step(self.bri, 'u'), 'l'), 'l')
                new_connects.append([self.step(self.bli, 'u'),self.bli])
                curr = self.bli
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'r')])
                    curr = self.step(curr, 'r')
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'u')])
                    curr = self.step(curr, 'u')
                return([child_block], new_connects)
            elif self.orientation == 'l':
                child_block = qhil_block(self.step(self.tli,'d'), self.step(self.step(self.tri, 'l'),'d'), self.bli, self.step(self.bri, 'l'), 'd')
                new_connects.append([self.step(self.bri, 'l'), self.bri])
                curr = self.bri
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'u')])
                    curr = self.step(curr, 'u')
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'l')])
                    curr = self.step(curr, 'l')
                return([child_block], new_connects)
            elif self.orientation == 'd':
                child_block = qhil_block(self.step(self.step(self.tli,'d'),'r'), self.step(self.tri,'d'), self.step(self.bli,'r'), self.bri, 'r')
                new_connects.append([self.step(self.tri, 'd'), self.tri])
                curr = self.tri
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'l')])
                    curr = self.step(curr, 'l')
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'd')])
                    curr = self.step(curr, 'd')
                return([child_block], new_connects)
            elif self.orientation == 'r':
                child_block = qhil_block(self.step(self.tli,'r'), self.tri, self.step(self.step(self.bli,'r'),'u'), self.step(self.bri,'u'), 'u')
                new_connects.append([self.step(self.tli, 'r'), self.tli])
                curr = self.tli
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'd')])
                    curr = self.step(curr, 'd')
                for _ in range(self.blocksize-1):
                    new_connects.append([curr, self.step(curr, 'r')])
                    curr = self.step(curr, 'r')
                return([child_block], new_connects)
        # Q split
        else:
            dis = int((self.blocksize/2)-1)
            if self.orientation == 'u':
                child_block_tl = qhil_block(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'l')
                child_block_tr = qhil_block((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'r')
                child_block_bl = qhil_block((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'u')
                child_block_br = qhil_block((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'u')
                new_connects.append((child_block_tl.bli,child_block_bl.tli))
                new_connects.append((child_block_bl.tri,child_block_br.tli))
                new_connects.append((child_block_br.tri,child_block_tr.bri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'l':
                child_block_tl = qhil_block(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'u')
                child_block_tr = qhil_block((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'l')
                child_block_bl = qhil_block((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'd')
                child_block_br = qhil_block((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'l')
                new_connects.append((child_block_bl.bri,child_block_br.bli))
                new_connects.append((child_block_br.tli,child_block_tr.bli))
                new_connects.append((child_block_tr.tli,child_block_tl.tri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'd':
                child_block_tl = qhil_block(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'd')
                child_block_tr = qhil_block((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'd')
                child_block_bl = qhil_block((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'l')
                child_block_br = qhil_block((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'r')
                new_connects.append((child_block_bl.tli,child_block_tl.bli))
                new_connects.append((child_block_tl.bri,child_block_tr.bli))
                new_connects.append((child_block_tr.bri,child_block_br.tri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'r':
                child_block_tl = qhil_block(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'r')
                child_block_tr = qhil_block((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'u')
                child_block_bl = qhil_block((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'r')
                child_block_br = qhil_block((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'd')
                new_connects.append((child_block_br.bli,child_block_bl.bri))
                new_connects.append((child_block_bl.tri,child_block_tl.bri))
                new_connects.append((child_block_tl.tri,child_block_tr.tli))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
    
    def __str__(self):
        return f'{self.tli},{self.tri},{self.bli},{self.bri} | {self.orientation}'
##

# make a Hilbert-L mapping for square grid of given size
def H_L(n):
    def complement(p,o):
        if p[0] == o:
            return p[1]
        else:
            return p[0]            
    connecs = []
    blocks = [qhil_block((0,0),(0,n-1),(n-1,0),(n-1,n-1),'u')]
    while not all([b.check_one() for b in blocks]):
        nblocks = []
        for bi in range(len(blocks)):
            block = blocks[bi]
            if not block.check_one():
                children, new_connecs = block.split()
                connecs = connecs + new_connecs
                nblocks = nblocks + children
        blocks = nblocks
    path = [(0,0)]
    curr = 0
    for _ in range(int(n**2-1)):
        curr += 1
        for pair in connecs:
            if path[-1] in pair and (x:=complement(pair, path[-1])) not in path:
                path.append(x)

    return {i:ltos(list(path[i])) for i in range(len(path))}


### code for generating a Hilbert-L curve with center optimized L blocks ### 
class qhil_block_adv:
    # Constructor method (initializes instance variables)
    # defines the bounds of the block(pair of list indices) and the orientation('u','l','r','d')
    # Takes an additional parameter for orientation of L blocks which can be either in state 0 or 1 which corresponds to their orientation eg, H_Lo of 6 will have 4 blocks so it can take [1,0,1,0]
    # Also takes a incrementing variable for the current L block
    def __init__(self, tli, tri, bli, bri, orientation, L_ors, L_i):
        # index of top right, top left, bottom right, bottom left
        self.tri = tri  
        self.tli = tli    
        self.bri = bri
        self.bli = bli
        self.L_ors =  L_ors
        self.L_i = L_i
        self.blocksize = 1 + abs(tri[1] - tli[1])
        self.orientation = orientation

    # Checks if a given block is of size 1
    def check_one(self):
        return self.tri==self.tli==self.bri==self.bli

    # takes a given pair and makes a step
    # eg given 
    def step(self, p, s):
        if s == 'u':
            return (p[0]-1, p[1])
        elif s == 'l':
            return (p[0], p[1]-1)
        elif s == 'd':
            return (p[0]+1,p[1])
        elif s == 'r':
            return (p[0], p[1]+1)

    # will tell you the total number of L blocks in the given block
    def Ls(self):
        power = 0
        accum = 0
        curr = self.blocksize
        while curr != 1:
            if curr % 2 == 1:
                curr -= 1
                accum += 4**(power)
            else:
                curr /= 2
                power += 1
        return accum


    # Split into quadrants, or make an L, returns the new children objects or object + L
    def split(self):
        # every split creates connections, this will store them and return it at the end
        new_connects = []
        # L split 
        if self.blocksize % 2 == 1:
            if self.orientation == 'u':
                if self.L_ors[self.L_i] == 0:
                    child_block = qhil_block_adv(self.tli, self.step(self.tri, 'l'), self.step(self.bli, 'u'), self.step(self.step(self.bri, 'u'), 'l'), 'l', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.bli, 'u'),self.bli])
                    curr = self.bli
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'r')])
                        curr = self.step(curr, 'r')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'u')])
                        curr = self.step(curr, 'u')
                    return([child_block], new_connects)
                else:
                    child_block = qhil_block_adv(self.step(self.tli,'r'), self.tri, self.step(self.step(self.bli, 'u'),'r'), self.step(self.bri, 'u'), 'r', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.bri, 'u'),self.bri])
                    curr = self.bri
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'l')])
                        curr = self.step(curr, 'l')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'u')])
                        curr = self.step(curr, 'u')
                    return([child_block], new_connects)
            elif self.orientation == 'l':
                if self.L_ors[self.L_i] == 0:
                    child_block = qhil_block_adv(self.step(self.tli,'d'), self.step(self.step(self.tri, 'l'),'d'), self.bli, self.step(self.bri, 'l'), 'd', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.bri, 'l'), self.bri])
                    curr = self.bri
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'u')])
                        curr = self.step(curr, 'u')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'l')])
                        curr = self.step(curr, 'l')
                    return([child_block], new_connects)
                else:
                    child_block = qhil_block_adv(self.tli, self.step(self.tri, 'l'), self.step(self.bli,'u'), self.step(self.step(self.bri, 'l'), 'u'), 'u', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.tri, 'l'), self.tri])
                    curr = self.tri
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'd')])
                        curr = self.step(curr, 'd')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'l')])
                        curr = self.step(curr, 'l')
                    return([child_block], new_connects)
            elif self.orientation == 'd':
                if self.L_ors[self.L_i] == 0:
                    child_block = qhil_block_adv(self.step(self.step(self.tli,'d'),'r'), self.step(self.tri,'d'), self.step(self.bli,'r'), self.bri, 'r', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.tri, 'd'), self.tri])
                    curr = self.tri
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'l')])
                        curr = self.step(curr, 'l')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'd')])
                        curr = self.step(curr, 'd')
                    return([child_block], new_connects)
                else:
                    child_block = qhil_block_adv(self.step(self.tli,'d'), self.step(self.step(self.tri,'d'),'l'), self.bli, self.step(self.bri,'l'), 'l', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.tli, 'd'), self.tli])
                    curr = self.tli
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'r')])
                        curr = self.step(curr, 'r')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'd')])
                        curr = self.step(curr, 'd')
                    return([child_block], new_connects)
            elif self.orientation == 'r':
                if self.L_ors[self.L_i] == 0:
                    child_block = qhil_block_adv(self.step(self.tli,'r'), self.tri, self.step(self.step(self.bli,'r'),'u'), self.step(self.bri,'u'), 'u', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.tli, 'r'), self.tli])
                    curr = self.tli
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'd')])
                        curr = self.step(curr, 'd')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'r')])
                        curr = self.step(curr, 'r')
                    return([child_block], new_connects)
                else:
                    child_block = qhil_block_adv(self.step(self.step(self.tli,'r'),'d'), self.step(self.tri,'d'), self.step(self.bli,'r'), self.bri, 'd', self.L_ors, self.L_i+1)
                    new_connects.append([self.step(self.bli, 'r'), self.bli])
                    curr = self.bli
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'u')])
                        curr = self.step(curr, 'u')
                    for _ in range(self.blocksize-1):
                        new_connects.append([curr, self.step(curr, 'r')])
                        curr = self.step(curr, 'r')
                    return([child_block], new_connects)
        # Q split
        else:
            dis = int((self.blocksize/2)-1)
            if self.orientation == 'u':
                child_block_tl = qhil_block_adv(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'l', self.L_ors, self.L_i)
                child_block_tr = qhil_block_adv((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'r', self.L_ors, self.L_i)
                child_block_bl = qhil_block_adv((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'u', self.L_ors, self.L_i)
                child_block_br = qhil_block_adv((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'u', self.L_ors, self.L_i)
                new_connects.append((child_block_tl.bli,child_block_bl.tli))
                new_connects.append((child_block_bl.tri,child_block_br.tli))
                new_connects.append((child_block_br.tri,child_block_tr.bri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'l':
                child_block_tl = qhil_block_adv(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'u', self.L_ors, self.L_i)
                child_block_tr = qhil_block_adv((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'l', self.L_ors, self.L_i)
                child_block_bl = qhil_block_adv((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'd', self.L_ors, self.L_i)
                child_block_br = qhil_block_adv((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'l', self.L_ors, self.L_i)
                new_connects.append((child_block_bl.bri,child_block_br.bli))
                new_connects.append((child_block_br.tli,child_block_tr.bli))
                new_connects.append((child_block_tr.tli,child_block_tl.tri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'd':
                child_block_tl = qhil_block_adv(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'd', self.L_ors, self.L_i)
                child_block_tr = qhil_block_adv((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'd', self.L_ors, self.L_i)
                child_block_bl = qhil_block_adv((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'l', self.L_ors, self.L_i)
                child_block_br = qhil_block_adv((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'r', self.L_ors, self.L_i)
                new_connects.append((child_block_bl.tli,child_block_tl.bli))
                new_connects.append((child_block_tl.bri,child_block_tr.bli))
                new_connects.append((child_block_tr.bri,child_block_br.tri))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
            elif self.orientation == 'r':
                child_block_tl = qhil_block_adv(self.tli, (self.tli[0], self.tli[1]+dis), (self.tli[0]+dis, self.tli[1]), (self.tli[0]+dis, self.tli[1]+dis), 'r', self.L_ors, self.L_i)
                child_block_tr = qhil_block_adv((self.tri[0], self.tri[1]-dis), self.tri, (self.tri[0]+dis, self.tri[1]-dis), (self.tri[0]+dis, self.tri[1]), 'u', self.L_ors, self.L_i)
                child_block_bl = qhil_block_adv((self.bli[0]-dis,self.bli[1]), (self.bli[0]-dis,self.bli[1]+dis), self.bli, (self.bli[0],self.bli[1]+dis), 'r', self.L_ors, self.L_i)
                child_block_br = qhil_block_adv((self.bri[0]-dis,self.bri[1]-dis), (self.bri[0]-dis,self.bri[1]), (self.bri[0],self.bri[1]-dis), self.bri, 'd', self.L_ors, self.L_i)
                new_connects.append((child_block_br.bli,child_block_bl.bri))
                new_connects.append((child_block_bl.tri,child_block_tl.bri))
                new_connects.append((child_block_tl.tri,child_block_tr.tli))
                return([child_block_tl, child_block_tr, child_block_bl, child_block_br], new_connects)
    
    def __str__(self):
        return f'{self.tli},{self.tri},{self.bli},{self.bri} | {self.orientation} | {self.L_i}, {self.L_ors}'
##

# make a general Hilbert-L mapping for square grid of given size given the L orientations
def H_Lo(n, L_ors):
    def complement(p,o):
        if p[0] == o:
            return p[1]
        else:
            return p[0]            
    connecs = []
    blocks = [qhil_block_adv((0,0),(0,n-1),(n-1,0),(n-1,n-1),'u', L_ors, 0)]
    while not all([b.check_one() for b in blocks]):
        nblocks = []
        for bi in range(len(blocks)):
            block = blocks[bi]
            if not block.check_one():
                children, new_connecs = block.split()
                connecs = connecs + new_connecs
                nblocks = nblocks + children
        blocks = nblocks
    path = [(0,0)]
    curr = 0
    for _ in range(int(n**2-1)):
        curr += 1
        for pair in connecs:
            if path[-1] in pair and (x:=complement(pair, path[-1])) not in path:
                path.append(x)

    return {i:ltos(list(path[i])) for i in range(len(path))}

### Greedy algorithm for square grid mapping, u=takes custom function for scoring ###
def G(n, score_func=lambda l: sum(score(l))):
    # checks a given point is in the grid and not in the existing list of visited points(l)
    def check(p, l):
        return (p[0] < n) and (p[0] > -1) and (p[1] < n) and (p[1] > -1) and (p not in l)
    mapp = {0:'[0,0]'}
    curr = [0,0]
    grid = []
    for i in range(n):
        for j in range(n):
            grid.append([i,j])
    visited = []
    for i in range(1,n**2):
        bscore = np.inf
        bp = None
        mappt = mapp
        for p in [item for item in grid if item not in visited and item != curr]:
            mappt[i] = ltos(p)
            if (s:=score_func(mappt)) < bscore:
                bscore = s
                bp = p
            mappt = mapp
        visited.append(curr)
        curr = bp
    mapp = {i:ltos(visited[i]) for i in range(n**2-1)}
    mapp[n**2-1] = ltos(curr)
    return mapp