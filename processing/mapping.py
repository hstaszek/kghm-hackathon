import logging

import pandas as pd

log = logging.getLogger('base')

common = {
    "Czas": ["Czas"],
    "LMAM_HC20?-_NDXSDI16-_PV": [
        "LMAM_HC201-_NDXSDI16-_PV",
        "LMAM_HC203-_NDXSDI16-_PV",
    ],
    "LMAM_HC20?-_NDXSDI8--_PV": [
        "LMAM_HC201-_NDXSDI8--_PV",
        "LMAM_HC203-_NDXSDI8--_PV",
    ],
    "LMAM_HC20?A_NDDT01---_TDI": [
        "LMAM_HC201A_NDDT01---_TDI",
        "LMAM_HC203A_NDDT01---_TDI",
    ],
    "LMAM_HC20?A_NDPT01---_TPI": [
        "LMAM_HC201A_NDPT01---_TPI",
        "LMAM_HC203A_NDPT01---_TPI",
    ],
    "LMAM_HC20?A_NDXRPI1--_PV": [
        "LMAM_HC201A_NDXRPI1--_PV",
        "LMAM_HC203A_NDXRPI1--_PV",
    ],
    "LMAM_HC20?A_WPFT01---_TFI": [
        "LMAM_HC201A_WPFT01---_TFI",
        "LMAM_HC203A_WPFT01---_TFI",
    ],
    "LMAM_HC20?A_WPXRFI1--_PV": [
        "LMAM_HC201A_WPXRFI1--_PV",
        "LMAM_HC203A_WPXRFI1--_PV",
    ],
    "LMAM_HC20?B_NDDT01---_TDI": [
        "LMAM_HC201B_NDDT01---_TDI",
        "LMAM_HC203B_NDDT01---_TDI",
    ],
    "LMAM_HC20?B_NDFT01---_TFI": [
        "LMAM_HC201B_NDFT01---_TFI",
        "LMAM_HC203B_NDFT01---_TFI",
    ],
    "LMAM_HC20?B_NDPT01---_TPI": [
        "LMAM_HC201B_NDPT01---_TPI",
        "LMAM_HC203B_NDPT01---_TPI",
    ],
    "LMAM_HC20?B_NDXRDI1--_PV": [
        "LMAM_HC201B_NDXRDI1--_PV",
        "LMAM_HC203B_NDXRDI1--_PV",
    ],
    "LMAM_HC20?B_NDXRFI1--_PV": [
        "LMAM_HC201B_NDXRFI1--_PV",
        "LMAM_HC203B_NDXRFI1--_PV",
    ],
    "LMAM_HC20?B_NDXRPI1--_PV": [
        "LMAM_HC201B_NDXRPI1--_PV",
        "LMAM_HC203B_NDXRPI1--_PV",
    ],
    "LMAM_HC20?B_WPFT01---_TFI": [
        "LMAM_HC201B_WPFT01---_TFI",
        "LMAM_HC203B_WPFT01---_TFI",
    ],
    "LMAM_HC20?B_WPXRFI1--_PV": [
        "LMAM_HC201B_WPXRFI1--_PV",
        "LMAM_HC203B_WPXRFI1--_PV",
    ],
    "LMAM_HC2?2A_NDDT01---_TDI": [
        "LMAM_HC212A_NDDT01---_TDI",
        "LMAM_HC232A_NDDT01---_TDI",
    ],
    "LMAM_HC2?2A_NDFT01---_TFI": [
        "LMAM_HC212A_NDFT01---_TFI",
        "LMAM_HC232A_NDFT01---_TFI",
    ],
    "LMAM_HC2?2A_NDPT01---_TPI": [
        "LMAM_HC212A_NDPT01---_TPI",
        "LMAM_HC232A_NDPT01---_TPI",
    ],
    "LMAM_HC2?2A_NDXRDI1--_PV": [
        "LMAM_HC212A_NDXRDI1--_PV",
        "LMAM_HC232A_NDXRDI1--_PV",
    ],
    "LMAM_HC2?2A_NDXRFI1--_PV": [
        "LMAM_HC212A_NDXRFI1--_PV",
        "LMAM_HC232A_NDXRFI1--_PV",
    ],
    "LMAM_HC2?2A_NDXRPI1--_PV": [
        "LMAM_HC212A_NDXRPI1--_PV",
        "LMAM_HC232A_NDXRPI1--_PV",
    ],
    "LMAM_HC2?2A_WPFT01---_TFI": [
        "LMAM_HC212A_WPFT01---_TFI",
        "LMAM_HC232A_WPFT01---_TFI",
    ],
    "LMAM_HC2?2A_WPXRFI1--_PV": [
        "LMAM_HC212A_WPXRFI1--_PV",
        "LMAM_HC232A_WPXRFI1--_PV",
    ],
    "LMAM_HC2?2B_NDDT01---_TDI": [
        "LMAM_HC212B_NDDT01---_TDI",
        "LMAM_HC232B_NDDT01---_TDI",
    ],
    "LMAM_HC2?2B_NDFT01---_TFI": [
        "LMAM_HC212B_NDFT01---_TFI",
        "LMAM_HC232B_NDFT01---_TFI",
    ],
    "LMAM_HC2?2B_NDPT01---_TPI": [
        "LMAM_HC212B_NDPT01---_TPI",
        "LMAM_HC232B_NDPT01---_TPI",
    ],
    "LMAM_HC2?2B_NDXRPI1--_PV": [
        "LMAM_HC212B_NDXRPI1--_PV",
        "LMAM_HC232B_NDXRPI1--_PV",
    ],
    "LMAM_HC2?2B_PL-------_TPS": [
        "LMAM_HC212B_PL-------_TPS",
        "LMAM_HC232B_PL-------_TPS",
    ],
    "LMAM_HC2?2B_PLDT01---_TDI": [
        "LMAM_HC212B_PLDT01---_TDI",
        "LMAM_HC232B_PLDT01---_TDI",
    ],
    "LMAM_HC2?2B_PLKL90---_TPG": [
        "LMAM_HC212B_PLKL90---_TPG",
        "LMAM_HC232B_PLKL90---_TPG",
    ],
    "LMAM_HC2?2B_WL-------_TAN": [
        "LMAM_HC212B_WL-------_TAN",
        "LMAM_HC232B_WL-------_TAN",
    ],
    "LMAM_HC2?2B_WPXRFI1--_PV": [
        "LMAM_HC212B_WPXRFI1--_PV",
        "LMAM_HC232B_WPXRFI1--_PV",
    ],
    "LMAM_HC251B_NDDT01---_TDI": [
        "LMAM_HC251B_NDDT01---_TDI",
        "LMAM_HC251B_NDDT01---_TDI",
    ],
    "LMAM_HC251B_NDFT01---_TFI": [
        "LMAM_HC251B_NDFT01---_TFI",
        "LMAM_HC251B_NDFT01---_TFI",
    ],
    "LMAM_HC251B_NDXSDI8--_PV": [
        "LMAM_HC251B_NDXSDI8--_PV",
        "LMAM_HC251B_NDXSDI8--_PV",
    ],
    "LMAM_HC251C_NDDT01---_TDI": [
        "LMAM_HC251C_NDDT01---_TDI",
        "LMAM_HC251C_NDDT01---_TDI",
    ],
    "LMAM_HC251C_NDFT01---_TFI": [
        "LMAM_HC251C_NDFT01---_TFI",
        "LMAM_HC251C_NDFT01---_TFI",
    ],
    "LMAM_HC251C_NDPT01---_TPI": [
        "LMAM_HC251C_NDPT01---_TPI",
        "LMAM_HC251C_NDPT01---_TPI",
    ],
    "LMAM_HC251C_NDXRDI1--_PV": [
        "LMAM_HC251C_NDXRDI1--_PV",
        "LMAM_HC251C_NDXRDI1--_PV",
    ],
    "LMAM_HC251C_NDXRFI1--_PV": [
        "LMAM_HC251C_NDXRFI1--_PV",
        "LMAM_HC251C_NDXRFI1--_PV",
    ],
    "LMAM_HC251C_NDXRPI1--_PV": [
        "LMAM_HC251C_NDXRPI1--_PV",
        "LMAM_HC251C_NDXRPI1--_PV",
    ],
    "LMAM_HC251C_WPFT01---_TFI": [
        "LMAM_HC251C_WPFT01---_TFI",
        "LMAM_HC251C_WPFT01---_TFI",
    ],
    "LMAM_HC251C_WPXRFI1--_PV": [
        "LMAM_HC251C_WPXRFI1--_PV",
        "LMAM_HC251C_WPXRFI1--_PV",
    ],
    "LMAM_K2?1--_NDXSDI8--_PV": [
        "LMAM_K211--_NDXSDI8--_PV",
        "LMAM_K231--_NDXSDI8--_PV",
    ],
    "LMAM_K2?1--_PLDT01---_TDI": [
        "LMAM_K211--_PLDT01---_TDI",
        "LMAM_K231--_PLDT01---_TDI",
    ],
    "LMAM_K2?1--_PPSKN01--_TLI": [
        "LMAM_K211--_PPSKN01--_TLI",
        "LMAM_K231--_PPSKN01--_TLI",
    ],
    "LMAM_K2?1--_WPFT01---_TFI": [
        "LMAM_K211--_WPFT01---_TFI",
        "LMAM_K231--_WPFT01---_TFI",
    ],
    "LMAM_K2?1--_WPXRFI1--_PV": [
        "LMAM_K211--_WPXRFI1--_PV",
        "LMAM_K231--_WPXRFI1--_PV",
    ],
    "LMAM_MF20?-_NDXSLI8--_PV": [
        "LMAM_MF201-_NDXSLI8--_PV",
        "LMAM_MF203-_NDXSLI8--_PV",
    ],
    "LMAM_MF21?-_NDXSLI8--_PV": [
        "LMAM_MF211-_NDXSLI8--_PV",
        "LMAM_MF213-_NDXSLI8--_PV",
    ],
    "LMAM_MK2?2-_---------_EPX": [
        "LMAM_MK212-_---------_EPX",
        "LMAM_MK232-_---------_EPX",
    ],
    "LMAM_MK2?2-_WPFT01---_TFI": [
        "LMAM_MK212-_WPFT01---_TFI",
        "LMAM_MK232-_WPFT01---_TFI",
    ],
    "LMAM_MK2?2-_WPXRFI1--_PV": [
        "LMAM_MK212-_WPXRFI1--_PV",
        "LMAM_MK232-_WPXRFI1--_PV",
    ],
    "LMAM_MP2?1-_---------_EPX": [
        "LMAM_MP211-_---------_EPX",
        "LMAM_MP231-_---------_EPX",
    ],
    "LMAM_MP2?1-_RDXRFI1--_PV": [
        "LMAM_MP211-_RDXRFI1--_PV",
        "LMAM_MP231-_RDXRFI1--_PV",
    ],
    "LMAM_MP2?1-_RDXRFI2--_PV": [
        "LMAM_MP211-_RDXRFI2--_PV",
        "LMAM_MP231-_RDXRFI2--_PV",
    ],
    "LMAM_MP2?1-_WLEQ01---_TDI": [
        "LMAM_MP211-_WLEQ01---_TDI",
        "LMAM_MP231-_WLEQ01---_TDI",
    ],
    "LMAM_MP2?1-_WPFT01---_TFI": [
        "LMAM_MP211-_WPFT01---_TFI",
        "LMAM_MP231-_WPFT01---_TFI",
    ],
    "LMAM_MP2?1-_WPXRFI2--_PV": [
        "LMAM_MP211-_WPXRFI2--_PV",
        "LMAM_MP231-_WPXRFI2--_PV",
    ],
    "LMAM_PM20?A_---------_EPX": [
        "LMAM_PM201A_---------_EPX",
        "LMAM_PM203A_---------_EPX",
    ],
    "LMAM_PM20?A_---------_TNI": [
        "LMAM_PM201A_---------_TNI",
        "LMAM_PM203A_---------_TNI",
    ],
    "LMAM_PM20?B_---------_EPX": [
        "LMAM_PM201B_---------_EPX",
        "LMAM_PM203B_---------_EPX",
    ],
    "LMAM_PM20?B_---------_TNI_ERPM": [
        "LMAM_PM201B_---------_TNI_ERPM",
        "LMAM_PM203B_---------_TNI_ERPM"
    ],
    "LMAM_PM20?B_--FAL1---_EI3": [
        "LMAM_PM201B_--FAL1---_EI3",
        "LMAM_PM203B_--FAL1---_EI3",
    ],
    "LMAM_PM20?B_--FAL1---_TNI_ERPM": [
        "LMAM_PM201B_--FAL1---_TNI_ERPM",
        "LMAM_PM203B_--FAL1---_TNI_ERPM",
    ],
    "LMAM_PM20?B_--FAL2---_EI3": [
        "LMAM_PM201B_--FAL2---_EI3",
        "LMAM_PM203B_--FAL2---_EI3",
    ],
    "LMAM_PM20?B_--FAL2---_TNI_ERPM": [
        "LMAM_PM201B_--FAL2---_TNI_ERPM",
        "LMAM_PM203B_--FAL2---_TNI_ERPM",
    ],
    "LMAM_PM2?1A_---------_EPX": [
        "LMAM_PM211A_---------_EPX",
        "LMAM_PM231A_---------_EPX",
    ],
    "LMAM_PM2?1B_---------_EPX": [
        "LMAM_PM211B_---------_EPX",
        "LMAM_PM231B_---------_EPX",
    ],
    "LMAM_PM2?2A_---------_EPX": [
        "LMAM_PM212A_---------_EPX",
        "LMAM_PM232A_---------_EPX",
    ],
    "LMAM_PM2?2A_---------_TNI": [
        "LMAM_PM212A_---------_TNI",
        "LMAM_PM232A_---------_TNI",
    ],
    "LMAM_PM2?2A_--FAL1---_EI3": [
        "LMAM_PM212A_--FAL1---_EI3",
        "LMAM_PM232A_--FAL1---_EI3",
    ],
    "LMAM_PM2?2A_--FAL1---_TNI_ERPM": [
        "LMAM_PM212A_--FAL1---_TNI_ERPM",
        "LMAM_PM232A_--FAL1---_TNI_ERPM",
    ],
    "LMAM_PM2?2A_--FAL2---_EI3": [
        "LMAM_PM212A_--FAL2---_EI3",
        "LMAM_PM232A_--FAL2---_EI3",
    ],
    "LMAM_PM2?2A_--FAL2---_TNI_ERPM": [
        "LMAM_PM212A_--FAL2---_TNI_ERPM",
        "LMAM_PM232A_--FAL2---_TNI_ERPM",
    ],
    "LMAM_PM2?2B_---------_EPX": [
        "LMAM_PM212B_---------_EPX",
        "LMAM_PM232B_---------_EPX",
    ],
    "LMAM_PM2?2B_---------_TNI": [
        "LMAM_PM212B_---------_TNI",
        "LMAM_PM232B_---------_TNI",
    ],
    "LMAM_PM251B_---------_EPX": [
        "LMAM_PM251B_---------_EPX",
        "LMAM_PM251B_---------_EPX",
    ],
    "LMAM_PM251C_---------_EPX": [
        "LMAM_PM251C_---------_EPX",
        "LMAM_PM251C_---------_EPX",
    ],
    "LMAM_PM251C_--FAL1---_EI3": [
        "LMAM_PM251C_--FAL1---_EI3",
        "LMAM_PM251C_--FAL1---_EI3",
    ],
    "LMAM_PM251C_--FAL1---_TNI_ERPM": [
        "LMAM_PM251C_--FAL1---_TNI_ERPM",
        "LMAM_PM251C_--FAL1---_TNI_ERPM",
    ],
    "LMAM_PM251C_--FAL2---_EI3": [
        "LMAM_PM251C_--FAL2---_EI3",
        "LMAM_PM251C_--FAL2---_EI3",
    ],
    "LMAM_PM251C_--FAL2---_TNI_ERPM": [
        "LMAM_PM251C_--FAL2---_TNI_ERPM",
        "LMAM_PM251C_--FAL2---_TNI_ERPM",
    ],
    "LMAM_RZM20?_NDLT01---_TLI": [
        "LMAM_RZM201_NDLT01---_TLI",
        "LMAM_RZM203_NDLT01---_TLI",
    ],
    "LMAM_RZM20?_NDXRLI1--_PV": [
        "LMAM_RZM201_NDXRLI1--_PV",
        "LMAM_RZM203_NDXRLI1--_PV",
    ],
    "LMAM_RZM20?_NDXSLI16-_PV": [
        "LMAM_RZM201_NDXSLI16-_PV",
        "LMAM_RZM203_NDXSLI16-_PV",
    ],
    "LMAM_RZM20?_NDXSLI8--_PV": [
        "LMAM_RZM201_NDXSLI8--_PV",
        "LMAM_RZM203_NDXSLI8--_PV",
    ],
    "LMAM_RZM20?_WPFT01---_TFI": [
        "LMAM_RZM201_WPFT01---_TFI",
        "LMAM_RZM203_WPFT01---_TFI",
    ],
    "LMAM_RZM20?_WPXRFI1--_PV": [
        "LMAM_RZM201_WPXRFI1--_PV",
        "LMAM_RZM203_WPXRFI1--_PV",
    ],
    "LMAM_RZM2?1_NDLT01---_TLI": [
        "LMAM_RZM211_NDLT01---_TLI",
        "LMAM_RZM231_NDLT01---_TLI",
    ],
    "LMAM_RZM2?1_NDXSLI16-_PV": [
        "LMAM_RZM211_NDXSLI16-_PV",
        "LMAM_RZM231_NDXSLI16-_PV",
    ],
    "LMAM_RZM2?1_NDXSLI8--_PV": [
        "LMAM_RZM211_NDXSLI8--_PV",
        "LMAM_RZM231_NDXSLI8--_PV",
    ],
    "LMAM_RZM2?1_OPPDO1---_TFI": [
        "LMAM_RZM211_OPPDO1---_TFI",
        "LMAM_RZM231_OPPDO1---_TFI",
    ],
    "LMAM_RZM2?1_OPXRFI2--_PV": [
        "LMAM_RZM211_OPXRFI2--_PV",
        "LMAM_RZM231_OPXRFI2--_PV",
    ],
    "LMAM_RZM2?1_OZPDO1---_TFI": [
        "LMAM_RZM211_OZPDO1---_TFI",
        "LMAM_RZM231_OZPDO1---_TFI",
    ],
    "LMAM_RZM2?1_OZXRFI1--_PV": [
        "LMAM_RZM211_OZXRFI1--_PV",
        "LMAM_RZM231_OZXRFI1--_PV",
    ],
    "LMAM_RZM2?2_NDLT01---_TLI": [
        "LMAM_RZM212_NDLT01---_TLI",
        "LMAM_RZM232_NDLT01---_TLI",
    ],
    "LMAM_RZM2?2_NDXRLI1--_PV": [
        "LMAM_RZM212_NDXRLI1--_PV",
        "LMAM_RZM232_NDXRLI1--_PV",
    ],
    "LMAM_RZM2?2_NDXSLI16-_PV": [
        "LMAM_RZM212_NDXSLI16-_PV",
        "LMAM_RZM232_NDXSLI16-_PV",
    ],
    "LMAM_RZM2?2_NDXSLI8--_PV": [
        "LMAM_RZM212_NDXSLI8--_PV",
        "LMAM_RZM232_NDXSLI8--_PV",
    ],
    "LMAM_RZM2?2_WPFT01---_TFI": [
        "LMAM_RZM212_WPFT01---_TFI",
        "LMAM_RZM232_WPFT01---_TFI",
    ],
    "LMAM_RZM2?2_WPXRFI1--_PV": [
        "LMAM_RZM212_WPXRFI1--_PV",
        "LMAM_RZM232_WPXRFI1--_PV",
    ],
    "LMAM_RZM251_NDLT01---_TLI": [
        "LMAM_RZM251_NDLT01---_TLI",
        "LMAM_RZM251_NDLT01---_TLI",
    ],
    "LMAM_RZM251_NDXRLI1--_PV": [
        "LMAM_RZM251_NDXRLI1--_PV",
        "LMAM_RZM251_NDXRLI1--_PV",
    ],
    "LMAM_RZM251_WPFT01---_TFI": [
        "LMAM_RZM251_WPFT01---_TFI",
        "LMAM_RZM251_WPFT01---_TFI",
    ],
    "LMAM_RZM251_WPXRFI1--_PV": [
        "LMAM_RZM251_WPXRFI1--_PV",
        "LMAM_RZM251_WPXRFI1--_PV",
    ],
    "LMAM_TM2?1-_---------_EPX": [
        "LMAM_TM211-_---------_EPX",
        "LMAM_TM231-_---------_EPX",
    ],
    "LMAM_TM2?1-_RDFQ01---_TFI": [
        "LMAM_TM211-_RDFQ01---_TFI",
        "LMAM_TM231-_RDFQ01---_TFI",
    ],
    "LMAM_TM2?2-_---------_EPX": [
        "LMAM_TM212-_---------_EPX",
        "LMAM_TM232-_---------_EPX",
    ],
    "LMAM_TM2?3-_---------_EPX": [
        "LMAM_TM213-_---------_EPX",
        "LMAM_TM233-_---------_EPX",
    ],
    "LMAY_WM22?2_--FAL----_CVF": [
        "LMAY_WM2212_--FAL----_CVF",
        "LMAY_WM2232_--FAL----_CVF",
    ],
    "LMAY_WM22?3_--FAL----_CVF": [
        "LMAY_WM2213_--FAL----_CVF",
        "LMAY_WM2233_--FAL----_CVF",
    ],
    "LMAY_WM32?2_--FAL----_CVF": [
        "LMAY_WM3212_--FAL----_CVF",
        "LMAY_WM3232_--FAL----_CVF",
    ],
    "LMAY_WM?2?3_--FAL----_CVF": [
        "LMAY_WM5213_--FAL----_CVF",
        "LMAY_WM3233_--FAL----_CVF",
    ],
    "LMAM_HC20??_PL-------_TPS": [
        "LMAM_HC201A_PL-------_TPS",
        "LMAM_HC203B_PL-------_TPS"
    ],
    "LMAM_HC20??_PLDT01---_TDI": [
        "LMAM_HC201A_PLDT01---_TDI",
        "LMAM_HC203B_PLDT01---_TDI"
    ],
    "LMAM_HC20??_PLKL90---_TPG": [
        "LMAM_HC201A_PLKL90---_TPG",
        "LMAM_HC203B_PLKL90---_TPG"
    ],
    "LMAM_OBM1C2_WPPT01---_TPI": [
        "LMAM_OBM1C2_WPPT01---_TPI"
    ]
}


def apply_mapping(df: pd.DataFrame):
    mapper = {}
    for col in df.columns:
        for dev, group in common.items():
            if col in group:
                mapper[col] = dev
    return df.rename(columns=mapper)[[c for c in mapper.values()]]
