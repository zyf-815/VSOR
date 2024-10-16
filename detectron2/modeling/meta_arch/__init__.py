# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import META_ARCH_REGISTRY, build_model  # isort:skip


# import all the meta_arch, so they will be registered
from .rcnn import RankSaliencyNetwork

from .binary_salient_build import build_Salient_predict