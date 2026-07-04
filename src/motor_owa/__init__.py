"""motor_owa - Motor adaptativo de recomendacion v2 (CRISP-DM).

Modelo adaptativo de recomendacion para el diseno de portafolios de
inversion en renta variable bajo incertidumbre, mediante el operador OWA
y perfiles conductuales de riesgo (tesis doctoral, UNAL Manizales).

Version 2, independiente del paquete anterior ``owa-adaptive``. Novedades:

1. Anclaje matematico de los 8 perfiles: latente z ~ N(0,1), octiles,
   orness_k = Phi(z_k) = (2k-1)/16 (derivado, no asertado). Se conserva
   la parametrizacion del Articulo 2 (0.158-0.865) como alternativa.
2. Carteras coherentes por construccion: 8 carteras (una por perfil) con
   volatilidad objetivo monotona en el orness (mayor apetito de riesgo
   => mayor volatilidad => mayor retorno potencial).
3. Adaptacion en el espacio latente z: la sorpresa del inversor tras cada
   horizonte actualiza z (asimetria de aversion a la perdida, Tversky &
   Kahneman, 1992) y puede recategorizar el perfil.
4. Validacion interna segun anteproyecto: RMSE, MAE, MAPE, NDCG@k, MRR,
   consistencia ordinal, particion 70-20-10.
"""

__version__ = "2.1.0"

from .config import Anchors, EngineConfig
from .latent import (phi, phi_inv, octile_orness, octile_z, classify_z,
                     latent_from_dimensions)
from .owa import owa, orness, andness, validate_weights
from .quantifiers import rim_weights, exponent_for_orness, weights_for_orness
from .profiles import PROFILE_NAMES, DIMENSIONS, Profile, all_profiles, get_profile
from .portfolio import PortfolioBuilder, PortfolioResult
from .adaptive import InvestorState, surprise, update_latent, harvest_and_recalibrate
from .elicitation import (QUESTIONNAIRE, declared_z, scores_from_z,
                          simulate_declared_scores, emotional_gap_metrics)
from .validation import (rmse, mae, mape, ndcg_at_k, mrr, ordinal_consistency,
                         split_70_20_10, coherence_spearman)
