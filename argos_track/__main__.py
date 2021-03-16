# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2021-03-16 12:20 PM

import sys
import logging
import yaml
from PyQt5 import (
    QtWidgets as qw,
)

from argos_track.track import ArgosTracker


app = qw.QApplication(sys.argv)
win = ArgosTracker()
win.setMinimumSize(800, 600)
win.setWindowTitle('Argos - track animals in video')
win.showMaximized()
app.aboutToQuit.connect(win.cleanup)
win.show()
sys.exit(app.exec_())