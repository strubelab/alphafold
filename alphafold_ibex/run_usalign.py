"""
Functions to run the US-align algorithm.

Example command adn output:

(base) login510-22:/ibex/user/guzmanfj/py/usalign/test$ ../lib/USalign Q7EY69.pdb Q6YPG3.pdb -mol prot -mm 1 -ter 1

 ********************************************************************
 * US-align (Version 20230609)                                      *
 * Universal Structure Alignment of Proteins and Nucleic Acids      *
 * Reference: C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Nat Methods*
 * Please email comments and suggestions to zhang@zhanggroup.org    *
 ********************************************************************

Name of Structure_1: Q7EY69.pdb:A:B (to be superimposed onto Structure_2)
Name of Structure_2: Q6YPG3.pdb:A:B
Length of Structure_1: 159 residues
Length of Structure_2: 187 residues

Aligned length= 109, RMSD=   4.01, Seq_ID=n_identical/n_aligned= 0.624
TM-score= 0.51979 (normalized by length of Structure_1: L=159, d0=4.70)
TM-score= 0.45008 (normalized by length of Structure_2: L=187, d0=5.10)
(You should use TM-score normalized by length of the reference structure)

(":" denotes residue pairs of d < 5.0 Angstrom, "." denotes other aligned residues)
GPAPARFCVYYDGHLPATRVLLMYVRIGTTATITARGHEFEVEAKDQNCKVILTNGKQAPDWLAAEPY*MAASGIETGTKLY---------------ISNLD----------YGVSNEDIK-ELFSEVGHLKRFAVHFDGYGRP----------NG-TAEVVFTRRSDAIAALKRYNNVLLDGKA----------------MKI-EVIGSDLGL------------------------*
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*                            .:..           ...:..:.  .:.  .    ....                  .  ...            . .:.:::..::.                ... ..                               *
GPAPARFCVYYDGHLPATRVLLMYVRIGTTATITARGHEFEVEAKDQNCKVILTNGKQAPDWLAAEPY*-------------EPAAMRVYTVCDESKYLIV-RNVPSLGCGDDLANLFAT-YGPV--D----ECTP--------MDAEDCDPYTD-VFFI------------K-FSQVSNARFAKRKLDESVFLGNRLQVSYAPQFE-------SLLDTKEKLEVRRKEVLGRMKSSS*

#Total CPU time is  0.21 seconds

"""