import copy
import random


FLOAT_MAX = 1e100


class Point:
    __slots__ = ["x", "y", "group", "name", "preLabel"]

    def __init__(self, x=0, y=0, group=0, name="1", preLabel=0):
        self.x, self.y, self.name, self.preLabel, self.group = x, y, group, name, preLabel


def generatePoints(pointsNumber, features, names, preLabels):
    points = [Point() for _ in range(pointsNumber)]
    for i, point in enumerate(points):
        points[i].x = features[i][0]
        points[i].y = features[i][1]
        points[i].name = names[i]
        points[i].preLabel = preLabels[i]
    return points


def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)


def getNearestCenter(point, clusterCenterGroup):
    minIndex = point.group
    minDistance = FLOAT_MAX
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(point, center)
        if (distance < minDistance):
            minDistance = distance
            minIndex = index
    return (minIndex, minDistance)


def kMeansPlusPlus(points, clusterCenterGroup):
    clusterCenterGroup[0] = copy.copy(random.choice(points))
    distanceGroup = [0.0 for _ in range(len(points))]
    sum = 0.0
    for index in range(1, len(clusterCenterGroup)):
        for i, point in enumerate(points):
            distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]
            sum += distanceGroup[i]
        sum *= random.random()
        for i, distance in enumerate(distanceGroup):
            sum -= distance;
            if sum < 0:
                clusterCenterGroup[index] = copy.copy(points[i])
                break
    for point in points:
        point.group = getNearestCenter(point, clusterCenterGroup)[0]
    return


def kMeans(points, clusterCenterNumber):
    clusterCenterGroup = [Point() for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(points, clusterCenterGroup)
    clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
    tolerableError, currentError = 5.0, FLOAT_MAX
    count = 0
    while currentError >= tolerableError:
        count += 1
        countCenterNumber = [0 for _ in range(clusterCenterNumber)]
        currentCenterGroup = [Point() for _ in range(clusterCenterNumber)]
        for point in points:
            currentCenterGroup[point.group].x += point.x
            currentCenterGroup[point.group].y += point.y
            countCenterNumber[point.group] += 1
        for index, center in enumerate(currentCenterGroup):
            center.x /= countCenterNumber[index]
            center.y /= countCenterNumber[index]
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):
            singleTrace.append(currentCenterGroup[index])
            currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
            clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])
        for point in points:
            point.group = getNearestCenter(point, clusterCenterGroup)[0]
    return clusterCenterGroup, clusterCenterTrace