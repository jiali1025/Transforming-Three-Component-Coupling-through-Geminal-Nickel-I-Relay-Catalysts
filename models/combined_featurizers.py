import numpy as np, torch
from rdkit import Chem
from chemprop.data.datapoints import ReactionDatapoint, MoleculeDatapoint
from chemprop.featurizers import (
    CondensedGraphOfReactionFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.data.molgraph import MolGraph

#prepare featurizer
CGR_FZR = CondensedGraphOfReactionFeaturizer()
MOL_FZR = SimpleMoleculeMolGraphFeaturizer()

#def _make_empty_graph(fzr=MOL_FZR) -> MolGraph:
def _make_empty_graph(fzr) -> MolGraph:
    v = np.zeros((1, fzr.atom_fdim),  np.single)
    e = np.empty((0, fzr.bond_fdim), np.single)
    edge = np.empty((2, 0), int)
    rev  = np.empty((0,),  int)
    return MolGraph(v, e, edge, rev)

#EMPTY_GRAPH = _make_empty_graph()
EMPTY_GRAPH = _make_empty_graph(MOL_FZR)
EMPTY_CGR_GRAPH = _make_empty_graph(CGR_FZR)

def smi_to_molgraph(smi: str) -> MolGraph:
    try:
        mol_dp = MoleculeDatapoint.from_smi(smi)          # RDKit parse+sanitise
        return MOL_FZR(mol_dp.mol)
    except Exception:
        return EMPTY_GRAPH

def rxn_smi_to_cgr(smi: str) -> MolGraph:
    try:
        rdp = ReactionDatapoint.from_smi(smi)
        return CGR_FZR((rdp.rct, rdp.pdt))
    except Exception:
        #return EMPTY_GRAPH
        return EMPTY_CGR_GRAPH


