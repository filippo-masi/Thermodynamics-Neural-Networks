# [Thermodynamics-based Artificial Neural Networks](https://doi.org/10.1016/j.jmps.2020.104277)



We propose a new class of data-driven, physics-based, neural networks for constitutive modeling of strain rate independent processes at the material point level, which we define as Thermodynamics-based Artificial Neural Networks (TANNs). The two basic principles of thermodynamics are encoded in the network’s architecture by taking advantage of automatic differentiation to compute the numerical derivatives of a network with respect to its inputs. In this way, derivatives of the free-energy, the dissipation rate and their relation with the stress and internal state variables are hardwired in the architecture of TANNs. Consequently, our approach does not have to identify the underlying pattern of thermodynamic laws during training, reducing the need of large data-sets. Moreover the training is more efficient and robust, and the predictions more accurate. Finally and more important, the predictions remain thermodynamically consistent, even for unseen data. Based on these features, TANNs are a starting point for data-driven, physics-based constitutive modeling with neural networks.

We demonstrate the applicability of TANNs for modeling elasto-plastic materials. Strain hardening and softening are also considered. TANNs’ architecture is general, enabling applications to materials with different or more complex behavior, without any modification.




If you use this code, please cite the related papers:


  - Masi, Stefanou, Vannucci, and Maffi-Berthier (2020). "[Thermodynamics-based Artificial Neural Networks for constitutive modeling](https://doi.org/10.1016/j.jmps.2020.104277)". Journal of the Mechanics and Physics of Solids, 104277.
  
  - Masi, Stefanou, Vannucci, and Maffi-Berthier (2021). "[Material modeling via Thermodynamics-based Artificial Neural Networks](https://franknielsen.github.io/SPIG-LesHouches2020/Masi-SPIGL2020.pdf)". In: Barbaresco F., Nielsen F. (eds) Proceedings of SPIGL'20: Geometric Structures of Statistical Physics, Information Geometry, and Learning. Springer.



## Citation


    @article{masi2020thermodynamics,
     title={Thermodynamics-based Artificial Neural Networks for constitutive modeling},
     author={Masi, Filippo and Stefanou, Ioannis and Vannucci, Paolo and Maffi-Berthier, Victor},
     journal={Journal of the Mechanics and Physics of Solids},
     pages={104277},
     year={2020},
     publisher={Elsevier}
     }
     
     
    @InProceedings{material2021modeling,
     author={Masi, Filippo and Stefanou, Ioannis and Vannucci, Paolo and Maffi-Berthier, Victor},
     editor={Barbaresco, Frédéric and Nielsen, Frank},
     title={Material modeling via Thermodynamics-based Artificial Neural Networks},
     booktitle={Proceedings of SPIGL'20: Geometric Structures of Statistical Physics, Information Geometry, and Learning},
     year={2021},
     publisher={Springer}
     }

https://zenodo.org/badge/DOI/10.5281/zenodo.4482669.svg
