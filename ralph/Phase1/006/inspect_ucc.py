import tencirchem
import inspect
print("UCC signature:", inspect.signature(tencirchem.UCC))
print("\nUCCSD signature:", inspect.signature(tencirchem.UCCSD))
# Check if transform parameter exists
sig = inspect.signature(tencirchem.UCC)
for param in sig.parameters:
    print(param, sig.parameters[param])
# Also check UCC.from_integral
if hasattr(tencirchem.UCC, 'from_integral'):
    print("\nfrom_integral signature:", inspect.signature(tencirchem.UCC.from_integral))