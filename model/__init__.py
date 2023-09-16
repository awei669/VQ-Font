# generator
from .generator import Generator

# discriminator
from .discriminator import disc_builder

def generator_dispatch():
    return Generator

from .content_encoder import content_enc_builder
from .decoder import dec_builder