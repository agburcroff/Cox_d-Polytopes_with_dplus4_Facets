from itertools import combinations, permutations, product, chain
from math import factorial
import numpy as np
import sys
from wolframclient.evaluation import WolframLanguageSession
from copy import deepcopy


#Running this code requires access to a Wolfram Kernel of version 11.3 of higher
#Add the file path to the Kernel as an argument in the following line
session = WolframLanguageSession()
from wolframclient.language import wl

#Input the list of missing faces, with facets labeled by 0,1,...,d+k-1
LannerList = {}
#Input the list of edges which could possibly be high-weight
HighWeightListAll = {}
#Input the list of all dashed and high-weight edges in some order
SolveListAll = []
#Input a list of pairs s.t. the corresponding edge weight will be solved for in the minor obtained by deleting these rows/columns
#Any additional pairs will be used to further constrain the system of equations
SolveCompAll = []
#Input the number of facets
num_points = 8

Len5perms = tuple(permutations(range(5)))
Len4perms = tuple(permutations(range(4)))


unweighted_list = []
SolveSubmats = []
HighWeightList = {}
SolveList = []
SolveComp = []
LargeLanner = []
prism_edges = []
good_large_lan_list = []
subdet_index = -2
undet_edge = False
mat41 = np.array([(0, 1, 0, 0), (1, 0, 3, 0), (0, 3, 0, 1), (0, 0, 1, 0)])
mat42 = np.array([(0, 3, 0, 0), (3, 0, 1, 0), (0, 1, 0, 2), (0, 0, 2, 0)])
mat43 = np.array([(0, 3, 0, 0), (3, 0, 1, 0), (0, 1, 0, 3), (0, 0, 3, 0)])
mat44 = np.array([(0, 3, 0, 0), (3, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 0)])
mat45 = np.array([(0, 2, 0, 1), (2, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)])
mat46 = np.array([(0, 2, 0, 1), (2, 0, 1, 0), (0, 1, 0, 2), (1, 0, 2, 0)])
mat47 = np.array([(0, 3, 0, 1), (3, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)])
mat48 = np.array([(0, 3, 0, 1), (3, 0, 1, 0), (0, 1, 0, 2), (1, 0, 2, 0)])
mat49 = np.array([(0, 3, 0, 1), (3, 0, 1, 0), (0, 1, 0, 3), (1, 0, 3, 0)])
lan_mat_4_list = (mat41, mat42, mat43, mat44, mat45, mat46, mat47, mat48, mat49)
for mat in lan_mat_4_list:
    for i in range(4):
        mat[i][i] = 1
mat51 = np.array([(0, 3, 0, 0, 0), (3, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 1), (0, 0, 0, 1, 0)])
mat52 = np.array([(0, 3, 0, 0, 0), (3, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 2), (0, 0, 0, 2, 0)])
mat53 = np.array([(0, 3, 0, 0, 0), (3, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 3), (0, 0, 0, 3, 0)])
mat54 = np.array([(0, 3, 0, 0, 0), (3, 0, 1, 0, 0), (0, 1, 0, 1, 1), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0)])
mat55 = np.array([(0, 2, 0, 0, 1), (2, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 1), (1, 0, 0, 1, 0)])
lan_mat_5_list = (mat51, mat52, mat53, mat54, mat55)
for mat in lan_mat_5_list:
    for i in range(5):
        mat[i][i] = 1
A3 = np.array([(0, 1, 0), (1, 0, 1), (0, 1, 0)])
A4 = np.array([(0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (0, 0, 1, 0)])
A5 = np.array([(0, 1, 0, 0, 0), (1, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 1), (0, 0, 0, 1, 0)])

B3 = np.array([(0, 1, 0), (1, 0, 2), (0, 2, 0)])
B4 = np.array([(0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 2), (0, 0, 2, 0)])
B5 = np.array([(0, 1, 0, 0, 0), (1, 0, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 1, 0, 2), (0, 0, 0, 2, 0)])

D4 = np.array([(0, 1, 0, 0), (1, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 0)])

F4 = np.array([(0, 1, 0, 0), (1, 0, 2, 0), (0, 2, 0, 1), (0, 0, 1, 0)])

H3 = np.array([(0, 1, 0), (1, 0, 3), (0, 3, 0)])
H4 = np.array([(0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 3), (0, 0, 3, 0)])

ell_3_list = (A3, B3, H3)
ell_4_list = (A4, B4, D4, F4, H4)
for mat in ell_3_list:
    for i in range(3):
        mat[i][i] = 1
for mat in ell_4_list:
    for i in range(4):
        mat[i][i] = 1

#If there is a (d+k-2)x(d+k-2) minor involving no variables, return the pair of deleted rows/columns (we call this a determined subdet)
def find_determined_subgraph(solveList):
    global undet_edge
    for pair in combinations(range(num_points), 2):
        determined = True
        for undet in solveList:
            if ((str(pair[0])) not in undet) and ((str(pair[1])) not in undet):
                determined = False
        if determined:
            return pair
    for pair in combinations(range(num_points), 2):
        determined = True
        num_undet = 0
        temp_undet_edge = False
        for undet in solveList:
            if ((str(pair[0])) not in undet) and ((str(pair[1])) not in undet):
                determined = False
                num_undet += 1
                temp_undet_edge = undet
        if determined:
            return pair
        elif num_undet == 1:
            undet_edge = temp_undet_edge
            return pair
    return False


#Sort a list of edges so that edges incident to either of a fixed pair of vertices appear first
def sort_according_to_pair(this_list, pair):
    temp_list = this_list.copy()
    for b in range(len(this_list)):
        edge = temp_list[b]
        if str(pair[0]) in edge or str(pair[1]) in edge:
            this_list.insert(len(this_list) - 1, this_list.pop(this_list.index(edge)))


#Sort a list of edges so that edges incident to either of a fixed pair of vertices appear last
def sort_according_to_pair_rev(this_list, pair):
    temp_list = this_list.copy()
    for b in range(len(this_list)):
        edge = temp_list[b]
        if str(pair[0]) in edge or str(pair[1]) in edge:
            this_list.insert(0, this_list.pop(this_list.index(edge)))


#If there is a determined subdet, find at which point in the set of unweighted edges we have handled all edges in this minor
def get_subdet_check_index(pair):
    for b in range(len(unweighted_list)):
        if str(pair[0]) in unweighted_list[b] or str(pair[1]) in unweighted_list[b]:
            return b - 1
    return -2


#check whether a certain minor involving at most one unweighted edge can be set to 0
def check_subdet(weight_mx, pair, subHighWeightList):
    spair = ''.join(map(str, pair))
    to_mathematica_mx(weight_mx)
    zero_submat = 0
    if undet_edge == False:
        det_str = "Abs[Det[N[A[[" + str(get_set_comp_from_str(spair)) + ", " + str(get_set_comp_from_str(spair)) + "]]]]] < 0.00000000001"
        zero_submat = session.evaluate("If[" + det_str + ", 1, 0]")
    elif undet_edge in LannerList:
        det_str = "s = NSolve[Det[N[A[[" + str(get_set_comp_from_str(spair)) + ", " + str(get_set_comp_from_str(spair)) + "]]]] == 0 && x" + undet_edge + " < -1, x" + undet_edge + ", Reals]"
        session.evaluate(det_str)
        zero_submat = session.evaluate('Length[s]')
    elif undet_edge in HighWeightListAll:
        det_str = "s = NSolve[Det[N[A[[" + str(get_set_comp_from_str(spair)) + ", " + str(get_set_comp_from_str(spair)) + "]]]] == 0 && x" + undet_edge + " > -1 && x" + undet_edge + " < -Cos[Pi/5], x" + undet_edge + ", Reals]"
        session.evaluate(det_str)
        session.evaluate('t = s[[All, 1, 2]]')
        m = session.evaluate('Length[t]')
        for a in range(m):
            session.evaluate('x' + undet_edge + '= t[[' + str(a + 1) + ']]')
            zero_submat += int(session.evaluate('If[Abs[N[Pi/ArcCos[-x' + undet_edge + ']] - Round[N[Pi/ArcCos[-x' + undet_edge + ']]]] < 0.001, 1, 0]'))
        session.evaluate('x' + undet_edge + '=.')
    else:
        print("error")
    if zero_submat:
        return True
    else:
        return False


#find which vertices are adjacent to 3 dashed edges
def get_prism_vertices():
    prism_vertices = []
    for a in range(num_points):
        lancount = 0
        for lan_dia in LannerList:
            if len(lan_dia) == 2 and str(a) in lan_dia:
                lancount += 1
        if lancount >= 3:
            prism_vertices.append(a)
    return prism_vertices


#make a list of edges incident to at least one prism vertex
def set_prism_edges():
    global prism_edges
    prism_vertices = get_prism_vertices()
    for v in prism_vertices:
        for a in range(num_points):
            if a != v:
                potential_edge = ''.join(map(str, sorted([a, v])))
                if potential_edge not in prism_edges + list(LannerList):
                    prism_edges.append(potential_edge)


#given a sublist of 0,1,...,d+k-1, return its complement in this set
def complement(this_list):
    comp = [point for point in range(num_points) if point not in this_list]
    return comp


def get_set_comp_from_str(vertex_comp):
    vertex_set = set()
    for a in range(num_points):
        if str(a) not in vertex_comp:
            vertex_set.add(a + 1)
    return vertex_set

def get_list_from_str(str):
    list1 = []
    list1[:0] = str
    for a in range(len(list1)):
        list1[a] = int(list1[a])
    return list1

def set_SolveSubmats():
    global SolveSubmats
    SolveSubmats = []
    for a in range(len(SolveComp)):
        SolveSubmats.append(get_set_comp_from_str(SolveComp[a]))
    return SolveSubmats

def is_elliptic(weight_mx, vertex_list):
    # -1 denotes dashed edge
    if len(vertex_list) <= 2:
        return True
    else:
        if is_positive_definite(vertex_list):
            return True
    return False

def is_decomposable(submat):
    dim = len(submat)
    temp = np.full((dim,dim), -2, dtype=int)
    for per in permutations(range(dim)):
        for x in range(dim):
            for y in range(dim):
                temp[x][y] = submat[per[x]][per[y]]
        for firstblock in range(1,dim):
            allzero = True
            for i in range(firstblock):
                for j in range(firstblock, dim):
                    if temp[i][j] != 0:
                        allzero = False
            for i in range(firstblock, dim):
                for j in range(firstblock):
                    if temp[i][j] != 0:
                        allzero = False
            if allzero:
                return True
    return False

def list_contains_Lanner(this_list):
    for lan in LannerList:
        contains_this = True
        for vertex in lan:
            if not (int(vertex) in this_list):
                contains_this = False
                break
        if contains_this:
            return True
    return False

def check_valid_elliptic(weight_mx):
    combo_verts = []
    for a in range(3, num_points - 2):
        combo_verts += list(combinations(range(num_points), a))
    for subvert in combo_verts:
        if not list_contains_Lanner(subvert):
                if not is_elliptic(weight_mx, subvert):
                    return False
    return True


def are_matrices_isomorphic_less_than(matA, matB):
    for per in permutations(range(len(matA))):
        less_than = True
        for a in range(len(matA)):
            for b in range(len(matA)):
                if matA[per[a]][per[b]] > matB[a][b]:
                    less_than = False
        if less_than:
            return True
    return False


def get_unweighted_edges(this_weight_mx):
    unweighted_edges = list()
    for edge in combinations(range(num_points), 2):
        if not ''.join(map(str, edge)) in LannerList.union(HighWeightList):
            if this_weight_mx[edge[0]][edge[1]] == -1:
                unweighted_edges.append(''.join(map(str, edge)))
    return unweighted_edges

def is_compatible(mat, index, wtuple):
    vertices = LargeLanner[index]
    if wtuple[0] == 4:
        vertexlist = [get_list_from_str(vertices)[a] for a in Len4perms[wtuple[2]]]
        Lanmat = lan_mat_4_list[wtuple[1]]
    elif wtuple[0] == 5:
        vertexlist = [get_list_from_str(vertices)[a] for a in Len5perms[wtuple[2]]]
        Lanmat = lan_mat_5_list[wtuple[1]]
    i = 0
    for x in vertexlist:
        j = 0
        for y in vertexlist:
            if mat[x][y] != Lanmat[i][j] and mat[x][y] > -1:
                return False
            j += 1
        i += 1
    return True


def get_submatrix(this_mat, sublist):
    sublist = sorted(sublist)
    sublen = len(sublist)
    submat = np.zeros((sublen, sublen))
    for a in range(sublen):
        for b in range(sublen):
            submat[a][b] = this_mat[sublist[a]][sublist[b]]
    return submat


def list_contains_Lanner(this_list):
    for lan in LannerList:
        contains_this = True
        for vertex in lan:
            if not (int(vertex) in this_list):
                contains_this = False
                break
        if contains_this:
            return True
    return False


def get_dashed_edges(Lanlist):
    these_dashed_edges = set()
    for missing_face in Lanlist:
        if len(missing_face) == 2:
            these_dashed_edges.add(missing_face)
    return these_dashed_edges


total_dashed_edges = get_dashed_edges(LannerList)


def count_neg_1_sols(this_list):
    neg_count = 0
    for item in this_list:
        if item < -1:
            neg_count += 1
    return neg_count


def assign_edge(edge, this_mat, weight):
    this_mat[int(edge[0])][int(edge[1])] = weight
    this_mat[int(edge[1])][int(edge[0])] = weight
    return this_mat

def assign_subdiagram(mat, Lanmat, vertexlist):
    for a,b in product(range(len(vertexlist)), range(len(vertexlist))):
        mat[int(vertexlist[a])][int(vertexlist[b])] = Lanmat[a][b]
    return mat


def ensure_Lanner_connected(weightmat):
    all_connected = True
    for lan in LannerList:
        temp = LannerList.copy()
        temp.remove(lan)
        for otherlan in temp:
            connected = False
            for a in range(len(otherlan)):
                for b in range(len(lan)):
                    if weightmat[int(otherlan[a])][int(lan[b])] != 0:
                        connected = True
                        break
            if not connected:
                all_connected = False
                break
    del temp
    return all_connected

def subdiagram_contains_highly_weighted_edge(vertex_set):
    for edge in combinations(vertex_set,2):
        edgestr = ''.join(map(str, sorted(edge)))
        if edgestr in HighWeightList:
            return edge
    return False

def test_weight(this_weight_mx, new_edge, new_wt):
    edge_set = (int(new_edge[0]), int(new_edge[1]))
    point_set = complement(edge_set)
    for point in point_set:
        if this_weight_mx[edge_set[0], point] >= 0 and this_weight_mx[edge_set[1], point] >= 0:
            triangle = [edge_set[0], edge_set[1], point]
            trianglestring = ''.join(map(str, sorted(triangle)))
            if trianglestring in LannerList:
                if subdiagram_contains_highly_weighted_edge(triangle):
                    if new_wt == 0 and (this_weight_mx[edge_set[0]][point] == 0 or this_weight_mx[edge_set[1]][point] == 0):
                        return False
                else:
                    if 1 / (float(this_weight_mx[edge_set[0], point] + 2)) + 1 / (float(this_weight_mx[edge_set[1], point] + 2)) + 1 / (float(new_wt + 2)) >= 1:
                        return False
            else:
                if subdiagram_contains_highly_weighted_edge(triangle):
                    if new_wt > 0:
                        return False
                else:
                    if 1 / (float(this_weight_mx[edge_set[0], point] + 2)) + 1 / (float(this_weight_mx[edge_set[1], point] + 2)) + 1 / (float(new_wt + 2)) <= 1:
                        return False
    for pair in combinations(point_set, 2):
        if not list_contains_Lanner(pair + edge_set):
            num_nonempty_edges = 0
            for potential_edge in combinations(pair + edge_set, 2):
                if potential_edge == edge_set:
                    if new_wt > 0:
                        num_nonempty_edges += 1
                else:
                    if this_weight_mx[potential_edge[0]][potential_edge[1]] > 0:
                        num_nonempty_edges += 1
            if num_nonempty_edges >= 4:
                return False
            elif num_nonempty_edges == 3:
                if not can_extend_to_elliptic(sorted(pair + edge_set), weight_mx):
                    return False
    return True


def can_extend_to_elliptic(vertex_set, weight_mx):
    submat = get_submatrix(weight_mx, vertex_set)
    # -1 denotes dashed edge
    #Modify
    if subdiagram_contains_highly_weighted_edge(vertex_set):
        high_weight_edge = subdiagram_contains_highly_weighted_edge(vertex_set)
        for a in range(len(submat)):
            if weight_mx[high_weight_edge[0]][a] > 0 and a != high_weight_edge[0] and a != high_weight_edge[1]:
                return False
    else:
        if len(submat) <= 2:
            return True
        elif len(submat) == 3:
            for ellmat in ell_3_list:
                if are_matrices_isomorphic_less_than(submat, ellmat):
                    return True
        elif len(submat) == 4:
            for ellmat in ell_4_list:
                if are_matrices_isomorphic_less_than(submat, ellmat):
                    return True
    return False


def draw_mx(mx):
    sz = len(mx)
    print("\diagrama")
    for i in range(sz):
        for j in range(i, sz):
            if i != j:
                if mx[i][j] == 1:
                    print("\draw (", i + 1, ") -- (", j + 1, ");", sep='')
                elif mx[i][j] == 2:
                    print("\draw[double] (", i + 1, ") -- (", j + 1, ");", sep='')
                elif mx[i][j] == 3:
                    print("\draw[triple] (", i + 1, ") -- (", j + 1, ");", sep='')
                elif mx[i][j] <= -1:
                    print("\draw[dashed] (", i + 1, ") -- (", j + 1, ");", sep='')
                elif mx[i][j] > 3:
                    weight = int(session.evaluate('Round[Pi/ArcCos[-x' + str(i) + str(j) + ']]'))
                    print("\draw (", i + 1, ") -- node[pos = 0.5, fill = white, minimum size = 1mm, draw=white, inner sep = 0pt, outer sep = 0pt,  right] {", weight, "} ++ (", j + 1, ");", sep='')
                    print("\draw (", i + 1, ") -- (", j + 1, ");", sep='')
    print("\end{tikzpicture}")
    print("&", flush=True)


def is_decomposable(submat):
    dim = len(submat)
    visited_verts = {0}
    to_visit = [0]
    while len(to_visit):
        for a in range(dim):
            if submat[to_visit[0]][a]:
                if a not in visited_verts:
                    visited_verts.add(a)
                    to_visit.append(a)
                    if len(visited_verts) == dim:
                        return False
        to_visit.remove(to_visit[0])
    return True


def to_mathematica_mx(int_mat):
    # define a matrix of integers
    session.evaluate('x =.')
    mat_dim = len(int_mat)
    mx_str = "{"
    for i in range(mat_dim):
        mx_str += "{"
        for j in range(mat_dim):
            if i == j:
                mx_str += "1"
            elif int_mat[i][j] == 3.5:
                mx_str += "x" + str(min(i, j)) + str(max(i, j))
            elif int_mat[i][j] == 1:
                mx_str += "-1/2"
            elif int_mat[i][j] == 2:
                mx_str += "-Sqrt[2]/2"
            elif int_mat[i][j] == 3:
                mx_str += "-(1+Sqrt[5])/4"
            elif int_mat[i][j] == 0:
                mx_str += "0"
            elif int_mat[i][j] < -1:
                mx_str += str(int_mat[i][j])
            else:
                mx_str += "x" + str(min(i, j)) + str(max(i, j))
            if j < mat_dim - 1:
                mx_str += ","
        if i < mat_dim - 1:
            mx_str += "},"
        else:
            mx_str += "}"
    mx_str += "}"
    session.evaluate('A = ' + mx_str)
    return

def get_var_list_mat(var_input_str):
    start_index = 10
    var_list = []
    for a in range(len(SolveList)):
        var_list.append(var_input_str[start_index + (a)*12: start_index + (a)*12 + 2])
    return var_list

def check_submats_2(this_unweighted_edges, this_weight_mx, this_goodmats, this_badmats):
    solve_str = 's = Sort[Simplify[Quiet[NSolve[Det[A] == 0 &&'
    var_str = '{'
    for solve_index in range(len(SolveSubmats)):
        submat_str = str(SolveSubmats[solve_index])
        solve_str += 'Det[A[[' + submat_str + ', ' + submat_str + ']]] == 0'
        #print(session.evaluate('Det[A[[' + submat_str + ', ' + submat_str + ']]]'))
        if solve_index < len(SolveSubmats) - 1:
            solve_str += ' && '
    for dashed_index in range(len(this_unweighted_edges)):
        dashed_edge = SolveList[dashed_index]
        var_str += 'x' + dashed_edge
        if dashed_edge in HighWeightList:
            solve_str += ' && x' + dashed_edge + ' > -1 && x' + dashed_edge + ' < -Cos[Pi/5]'
        else:
            solve_str += ' && x' + dashed_edge + ' < -1'
        if dashed_index < len(this_unweighted_edges) - 1:
            var_str += ', '
    solve_str += ','
    var_str += '}'
    solve_str += var_str + ', Reals], NSolve::ratnz]]]'
    t = session.evaluate(solve_str)
    if str(t) == '((),)':
        print("%%%CHECK BY HAND%%%")
        print("troublesome edge: ", dashed_edge)
        print(weight_mx)
        return False
    else:
        session.evaluate('t = s[[All, All, 2]]')
        varlist = get_var_list_mat(str(session.evaluate('v = s[[All, All, 1]]')))
        m = session.evaluate('Length[s]')
        valid = False
        for a in range(m):
            valid = True
            for dashed_index in range(len(this_unweighted_edges)):
                dashed_edge = varlist[dashed_index]
                session.evaluate('x' + dashed_edge + '= t[[' + str(a + 1) + ']][[' + str(dashed_index + 1) + ']]')
                if dashed_edge in HighWeightList:
                    correct_range = int(session.evaluate('If[x' + dashed_edge + ' > -1 && x' + dashed_edge + ' < -Cos[Pi/5], 1, 0]'))
                    if correct_range:
                        integral = int(session.evaluate('If[Abs[N[Pi/ArcCos[-x' + dashed_edge + ']] - Round[N[Pi/ArcCos[-x' + dashed_edge + ']]]] < 0.001, 1, 0]'))
                    if not (correct_range and integral):
                        valid = False
                        break
                else:
                    correct_range = int(session.evaluate('If[x' + dashed_edge + ' < -1, 1, 0]'))
                    if not correct_range:
                        valid = False
                        break
            if valid:
                rank = int(session.evaluate('MatrixRank[A]'))
                if rank == num_points - 3:
                    #test if diagram is actually of this type
                    if not mx_isomorphic_to_mx_in_list_ignoring_dashed(this_weight_mx, this_goodmats + this_badmats):
                        if check_valid_elliptic(weight_mx):
                            print('%' + str(len(this_goodmats) + 1), flush=True)
                            print(weight_mx)
                            this_goodmats.append(this_weight_mx.copy())
                            if not endgame:
                                for edge in SolveList:
                                    if edge in HighWeightList:
                                        #print("x" + edge + ": " + str(session.evaluate('Pi/ArcCos[-x' + edge + ']')))
                                    else:
                                        #print("x" + edge + ": " + str(session.evaluate('x' + edge)))
                        else:
                            this_badmats.append(this_weight_mx.copy())
                            valid = False
                    else:
                        valid = False
                else:
                    valid = False
            for dashed_index in range(len(this_unweighted_edges)):
                dashed_edge = SolveList[dashed_index]
                session.evaluate('x' + dashed_edge + '= .')
    return valid


def assign_large_lanner_list():
    global LargeLanner
    LargeLanner = []
    for lan in LannerList:
        if len(lan) > 3:
            LargeLanner.append(lan)

def is_det_zero(this_mat, vertex_list):
    to_mathematica_mx(get_submatrix(this_mat, vertex_list))
    m = session.evaluate('MatrixRank[A]')
    if m < len(vertex_list):
        return True
    return False

def is_positive_definite(vertex_list):
    sub_vertex_str = str(set([x+1 for x in vertex_list]))
    m = session.evaluate('If[Min[Eigenvalues[N[A[[' + sub_vertex_str + ',' + sub_vertex_str + ']]]]] > 0, 1, 0]')
    if m:
        return True
    return False


def mx_isomorphic_to_mx_in_list_ignoring_dashed(this_mat, this_list):
    lenA = len(this_mat)
    for per in permutations(range(lenA)):
        for list_mat in this_list:
            all_equal = True
            for a, b in product(range(lenA), range(lenA)):
                if not this_mat[per[a]][per[b]] == list_mat[a][b]:
                    if this_mat[per[a]][per[b]] > -1 or list_mat[a][b] > -1:
                        all_equal = False
                        break
            if all_equal:
                return True
    return False


def add_minimal_amt_weight_list(weight_list, index, weight_mx):
    for i in reversed(range(index + 1)):
        if weight_list[i] < 3:
            weight_list[i] += 1
            assign_edge(unweighted_list[i], weight_mx, weight_list[i])
            for j in range(i + 1, len(weight_list)):
                weight_list[j] = -1
                assign_edge(unweighted_list[j], weight_mx, -1)
            return i
    return -1


def can_be_solved(this_weight_mx, vertex_list, edge):
    assign_edge(edge,this_weight_mx,-1)
    to_mathematica_mx(get_submatrix(this_weight_mx, vertex_list))
    session.evaluate('s = DeleteDuplicates[Simplify[Solve[x < -1 && Det[A] == 0, x, Reals]]]')
    session.evaluate('g = .')
    sols = session.evaluate('s[[All,1,2]]')
    m = session.evaluate('Length[s]')
    if m >= 1:
        return m, sols[0]
    return m, 0


def get_next_weighting(weight_list, start, weight_mx):
    if len(weight_list) == 0:
        return True
    if start:
        weight_list[0] = 0
        index = 0
        assign_edge(unweighted_list[index], weight_mx, 0)
    else:
        index = add_minimal_amt_weight_list(weight_list, len(weight_list) - 1, weight_mx)
    while index < len(weight_list) and index >= 0:
        changed = False
        # START Diagram-specific code
        # END Diagram-specific code
        if index == subdet_index + 1:
            if not check_subdet(weight_mx, pair, subHighWeightList):
                index = add_minimal_amt_weight_list(weight_list, index - 1, weight_mx)
                changed = True
        if not changed:
            if not (test_weight(weight_mx, unweighted_list[index], weight_list[index]) and ensure_Lanner_connected(weight_mx)):
                index = add_minimal_amt_weight_list(weight_list, index, weight_mx)
            else:
                index += 1
                if index < len(weight_list) and weight_list[index] == -1:
                    weight_list[index] = 0
                    assign_edge(unweighted_list[index], weight_mx, 0)
    if index >= 0:
        #print(weight_list)
        return True
    else:
        for a in range(len(weight_list)):
            weight_list[a] = 3
        return False

def reset_weight_mx(this_weight_mx):
    for a in range(num_points):
        for b in range(num_points):
            if a == b:
                this_weight_mx[a][b] = 1
            else:
                this_weight_mx[a][b] = -1
    for prism_edge in prism_edges:
        assign_edge(prism_edge, this_weight_mx, 0)
    return this_weight_mx

def reset_weight_mx_no_prism(this_weight_mx):
    for a in range(num_points):
        for b in range(num_points):
            if a == b:
                this_weight_mx[a][b] = 1
            else:
                this_weight_mx[a][b] = -1
    return this_weight_mx

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def set_SolveList_and_SolveComp():
    global SolveList
    global SolveComp
    SolveList = []
    SolveComp = []
    for a in range(len(SolveListAll)):
        if SolveListAll[a] in HighWeightList or SolveListAll[a] in LannerList:
            SolveList.append(SolveListAll[a])
    for b in range(len(SolveCompAll)):
        SolveComp.append(SolveCompAll[b])

def revert_mx_lan_list(index, large_lan_list, weight_mx):
    reset_weight_mx(weight_mx)
    for a in range(index):
        set_diagram(weight_mx, large_lan_list[a], a)

def revert_mx_lan_list_no_prism(index, large_lan_list, weight_mx):
    reset_weight_mx_no_prism(weight_mx)
    for a in range(index):
        set_diagram(weight_mx, large_lan_list[a], a)


def add_minimal_amt_large_lan_list(large_lan_list, index):
    added = False
    for a in reversed(range(index + 1)):
        if large_lan_list[a][2] < factorial(large_lan_list[a][0]) - 1:
            large_lan_list[a][2] = large_lan_list[a][2] + 1
            added = True
        elif large_lan_list[a][0] == 4 and large_lan_list[a][1] < len(lan_mat_4_list) - 1:
            large_lan_list[a][1] = large_lan_list[a][1] + 1
            large_lan_list[a][2] = 0
            added = True
        elif large_lan_list[a][0] == 5 and large_lan_list[a][1] < len(lan_mat_5_list) - 1:
            large_lan_list[a][1] = large_lan_list[a][1] + 1
            large_lan_list[a][2] = 0
            added = True
        if added:
            revert_mx_lan_list(a, large_lan_list, weight_mx)
            for j in range(a + 1, len(large_lan_list)):
                large_lan_list[j][1] = -1
                large_lan_list[j][2] = -1
            return a
    return -1

def set_diagram(weight_mx, wtuple, index):
    vertices = LargeLanner[index]
    if wtuple[0] == 4:
        assign_subdiagram(weight_mx, lan_mat_4_list[wtuple[1]], [get_list_from_str(vertices)[a] for a in Len4perms[wtuple[2]]])
    elif wtuple[0] == 5:
        assign_subdiagram(weight_mx, lan_mat_5_list[wtuple[1]], [get_list_from_str(vertices)[a] for a in Len5perms[wtuple[2]]])
    else:
        print("bad tuple")

#store as a list of triples, according to size of lanner diagram, which lanner diagram, and permutation of vertices.
def get_next_large_lan_list(large_lan_list, start, weight_mx):
    if start:
        reset_weight_mx(weight_mx)
        large_lan_list[0][1] = 0
        large_lan_list[0][2] = 0
        index = 0
        set_diagram(weight_mx, large_lan_list[0], index)
    else:
        index = add_minimal_amt_large_lan_list(large_lan_list, len(large_lan_list) - 1)
    while index < len(large_lan_list) and index >= 0:
        changed = False
        if not changed:
            if not is_compatible(weight_mx, index, large_lan_list[index]):
                index = add_minimal_amt_large_lan_list(large_lan_list, index)
            else:
                set_diagram(weight_mx, large_lan_list[index], index)
                index += 1
                if index < len(large_lan_list) and large_lan_list[index][1] == -1:
                    large_lan_list[index][1] = 0
                    large_lan_list[index][2] = 0
    if index >= 0:
        return True
    else:
        return False

def initialize_large_lan_list(large_lan_list):
    for a in range(len(LargeLanner)):
        large_lan_list.append([len(LargeLanner[a]), -1, -1])


def assess_diagram_after_large_lans(prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list, endgame):
    global good_large_lan_list
    for high_weight_edge in HighWeightList:
        assign_edge(high_weight_edge, weight_mx, 3.5)
    weight_list = [-1] * len(unweighted_list)
    get_next_weighting(weight_list, True, weight_mx)
    #weight_list = [0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,2,0,0,2]
    #weight_list = [0,0,0,0,0,1,1,0,0,0,0,0,0,0,2,0,2,0,1,1,0,0,1]
    for edge in range(len(weight_list)):
        assign_edge(unweighted_list[edge], weight_mx, weight_list[edge])
    while weight_list != [3] * len(unweighted_list):
        # if num_mats % 1 == 0:
        #     print("total:", num_mats, flush=True)
        #     print(weight_list, flush = True)
        num_mats += 1
        to_mathematica_mx(weight_mx)
        # check_mathematica_submats(SolveList, weight_mx, goodmats)
        if check_submats_2(SolveList, weight_mx, goodmats, badmats) and not endgame:
            templist = list()
            for a in range(len(large_lan_list)):
                templist.append(deepcopy(large_lan_list[a]))
            good_large_lan_list.append(deepcopy(templist))
        get_next_weighting(weight_list, False, weight_mx)
    return prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list

if __name__ == '__main__':
    #determine the Lanner subdiagrams of size at least 4
    assign_large_lanner_list()
    set_prism_edges()
    total_good = 0
    #Range over all choices of highly weighted edges
    for subHighWeightList in powerset(HighWeightListAll):
        if len(subHighWeightList) == len(HighWeightListAll):
            #reinitialize all high weight parameters
            HighWeightList = set(subHighWeightList)
            set_SolveList_and_SolveComp()
            set_SolveSubmats()
            subdet_index = -2
            undet_edge = False
            pair = find_determined_subgraph(SolveList)
            num_mats = 0
            num_good_mats = 0
            goodmats = list()
            badmats = list()
            good_large_lan_list = list()
            print("--------", HighWeightList, "--------", flush = True)
            #prepare to assign large lanner lists
            large_lan_list = []
            initialize_large_lan_list(large_lan_list)
            weight_mx = np.zeros((num_points, num_points))
            # reset weight matrix to -1's with 1's along diagonal
            reset_weight_mx(weight_mx)
            start = True
            prev_first_perm = -1
            endgame = False
            large_lan_list = list()
            if len(unweighted_list) == 0:
                unweighted_list = get_unweighted_edges(weight_mx)
                for hw_edge in HighWeightList:
                    sort_according_to_pair_rev(unweighted_list, hw_edge)
                if pair:
                    sort_according_to_pair(unweighted_list, pair)
                    subdet_index = get_subdet_check_index(pair)
            #print(unweighted_list)
            start = False
            prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list = assess_diagram_after_large_lans(prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list, endgame)
            endgame = True
            #print(len(goodmats), flush=True)
            total_good += len(goodmats)
            goodmats = list()
            for good_large_lan in good_large_lan_list:
                revert_mx_lan_list_no_prism(len(large_lan_list), good_large_lan, weight_mx)
                prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list = assess_diagram_after_large_lans(prev_first_perm, weight_mx, goodmats, badmats, num_mats, large_lan_list, endgame)
            #print(len(goodmats), flush=True)
            unweighted_list = []
    #print(total_good)
    sys.exit()