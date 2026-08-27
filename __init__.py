def classFactory(iface):
    from .coulomb_stress_transfer import CoulombStressTransferPlugin
    return CoulombStressTransferPlugin(iface)
