    """Placeholder surrogate module.
    Replace with code to load your real ANN (trained on CCD) and call ann.predict(state.reshape(1,-1))
    """


def load_dummy():
    class Dummy:
        def predict(self, state):
            t, f, p, s = state
            score = 0.6*(t-200)/(280-200) + 0.25*(s-30)/(50-30) - 0.15*(f-875)/(950-875) + 0.1*(p-1.6)/(2.0-1.6)
            score = max(0.0, min(1.0, score))
            return score * 100.0
    return Dummy()
