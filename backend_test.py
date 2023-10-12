import numpy as np
import sys
import traceback
import backend


class Tester:
    def __init__(self):
        self.module = None
        self.runtime = 300

    #############################################
    # Task a
    #############################################

    def testA(self, l: list, task):
        comments = ""

        def evaluate(A, b, reference, epsilon=1e-16):
            nonlocal comments
            try:
                x = self.module.solveLinearSystem(A, b)
                if ((np.abs(x - reference) < epsilon).all()):
                    comments += "passed."
                else:
                    comments += "failed."
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Identity
        comments += "Identity case "

        A = np.identity(30)
        b = np.ones(30)
        for i in range(30):
            b[i] = (i + 1)
        reference = np.copy(b)
        evaluate(A, b, reference)

        # 30x30 floats
        comments += "30x30 case "

        A = np.tril(np.ones((30, 30)))
        tmp = np.triu(np.ones((30, 30)))
        for i in range(30):
            A[i] *= i + 1
            A[:, i] /= i + 1
            tmp[i] *= (i + 1) ** 2
            tmp[:, i] /= (i + 1) ** 2
        A = A.dot(tmp)
        A *= np.pi / np.e
        b = np.ones(30)
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference, 1e-10)

        # 30x30 instable
        comments += "30x30 unstable case "

        A = np.tril(np.ones((30, 30)))
        tmp = np.triu(np.ones((30, 30)))
        for i in range(30):
            A[i] *= i + 1
            A[:, i] /= i + 1
            tmp[i] *= (i + 1) ** 2
            tmp[:, i] /= (i + 1) ** 2
        A = A.dot(tmp)
        A *= np.pi / np.e
        for i in range(30):
            A[i] *= np.exp(i)
            A[:, i] /= np.exp(i)
        b = np.ones(30)
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference, 1e-2)

        # 10x10 Pivoting
        comments += "10x10 Pivoting case "

        A = np.triu(np.ones((10, 10)))
        A = np.roll(A, 1, axis=0)
        b = np.ones(10)
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference)
        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task b
    #############################################

    def testB(self, l: list, task):
        comments = ""

        def evaluate(A, b, reference):
            nonlocal comments
            try:
                if (self.module.isConsistent(np.copy(A), np.copy(b)) == reference):
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # 10x10 upper triangular
        comments += "10x10 upper triangle case "

        A = np.triu(np.ones((10, 10)))
        b = np.ones(10)
        x = np.linalg.solve(A, b)
        reference = np.allclose(np.dot(A, x), b)
        evaluate(A, b, reference)

        # 10x10 floats
        comments += "10x10 case with floating numbers "

        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.
        A[-1] = A[0] + A[-2]
        b = np.ones(10)
        # Solve the linear system of equations
        x = np.linalg.solve(A, b)
        reference = np.allclose(np.dot(A, x), b)
        evaluate(A, b, reference)

        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task c
    #############################################

    def testC(self, l: list, task):
        comments = ""

        def evaluate(scanner):
            nonlocal comments
            try:
                A, b = self.module.setUpLinearSystem(scanner)
                x = np.linalg.lstsq(A, b.reshape(A.shape[0]), rcond=None)[0]
                if ((np.abs(x.reshape(scanner.resolution, scanner.resolution) - scanner.image) < 1e-3).all()):
                    comments += "passed."
                else:
                    comments += "failed."
            except Exception as e:
                comments += "crashed." + str(e) + ""
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + ""

        # Default 20x20
        comments += "Default case "


        # 30x30
        comments += "30x30 case "

        #scanner = CTScanner(30)
        #evaluate(scanner, 1)

        l.extend([task, comments])


    def performTest(self, func, task):
        # manager = multiprocessing.Manager()
        # localList = manager.list()
        # p = multiprocessing.Process(target=func, args=(localList,))
        # p.start()
        # p.join(self.runtime)
        # p.kill()
        # if (p.exitcode != 0):
        #     return []
        # else:
        #     return list(localList)

        l = []
        try:
            func(l, task)
            return l
        except Exception as e:
            return []

    def runTests(self, module, l):
        self.module = module

        def evaluateResult(task, result):
            if (len(result) == 0):
                l.append([task, 0, "Interrupt."])
            else:
                l.append(result)

        result = self.performTest(self.testA, '1.1a)')
        evaluateResult("1a)", result)

        result = self.performTest(self.testB, '1.1b)')
        evaluateResult("1b)", result)

        result = self.performTest(self.testC, '1.1c)')
        evaluateResult("1c)", result)

        return l


tester = Tester()
overall_result = []
tester.runTests(backend, overall_result)

