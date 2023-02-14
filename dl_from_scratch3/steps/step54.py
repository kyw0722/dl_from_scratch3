if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import test_mode
import dezero.functions as F

def main() :
    x = np.ones(5)
    print(x)

    y = F.dropout(x)
    print(y)

    with test_mode() :
        y = F.dropout(x)
        print(y)

if __name__ == '__main__' :
    main()