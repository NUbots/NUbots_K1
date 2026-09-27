import { FieldDimensions } from "../../../../../shared/field/dimensions";
import { Vector3 } from "../../../../../shared/math/vector3";

export interface FieldLandmark {
  position: Vector3;
  type: "L_INTERSECTION" | "T_INTERSECTION" | "X_INTERSECTION";
}

export interface FieldLandmarkGraph {
  landmarks: FieldLandmark[];
  /** Index pairs into `landmarks` that are directly connected by a straight line on the pitch. */
  edges: [number, number][];
}

/**
 * Mirrors `setup_field_landmarks` in shared/utility/localisation/FieldLineOccupanyMap.hpp, so the
 * landmarks here are the same fixed field-map points the robot's own localisation associates
 * detected intersections against (see `Field.association_lines`). The edges (not present on the
 * backend, which only needs points for its cost function) describe which landmarks are joined by
 * a straight pitch marking, so detected/associated points can be connected into a field outline.
 */
export function buildFieldLandmarkGraph(dim: FieldDimensions): FieldLandmarkGraph {
  const halfLength = dim.fieldLength * 0.5;
  const halfWidth = dim.fieldWidth * 0.5;
  const circleRadius = dim.centerCircleDiameter * 0.5;
  const halfPenaltyAreaWidth = dim.penaltyAreaWidth * 0.5;
  const halfGoalAreaWidth = dim.goalAreaWidth * 0.5;

  const landmarks: FieldLandmark[] = [];
  const addL = (x: number, y: number) => landmarks.push({ position: new Vector3(x, y, 0), type: "L_INTERSECTION" }) - 1;
  const addT = (x: number, y: number) => landmarks.push({ position: new Vector3(x, y, 0), type: "T_INTERSECTION" }) - 1;
  const addX = (x: number, y: number) => landmarks.push({ position: new Vector3(x, y, 0), type: "X_INTERSECTION" }) - 1;

  // Pitch corners.
  const cornerNegTop = addL(-halfLength, halfWidth);
  const cornerNegBot = addL(-halfLength, -halfWidth);
  const cornerPosTop = addL(halfLength, halfWidth);
  const cornerPosBot = addL(halfLength, -halfWidth);

  // Touchline midpoints.
  const touchTop = addT(0, halfWidth);
  const touchBot = addT(0, -halfWidth);

  // Centre line: touchline-to-touchline, split by where it crosses the centre circle.
  const centre = addX(0, 0);
  const circleTop = addX(0, circleRadius);
  const circleBot = addX(0, -circleRadius);

  const edges: [number, number][] = [
    [cornerNegTop, touchTop],
    [touchTop, cornerPosTop],
    [cornerNegBot, touchBot],
    [touchBot, cornerPosBot],
    [touchTop, circleTop],
    [circleTop, centre],
    [centre, circleBot],
    [circleBot, touchBot],
  ];

  // Goal area (6-yard box) T's, always present.
  const goalTPosTop = addT(halfLength, halfGoalAreaWidth);
  const goalTPosBot = addT(halfLength, -halfGoalAreaWidth);
  const goalTNegTop = addT(-halfLength, halfGoalAreaWidth);
  const goalTNegBot = addT(-halfLength, -halfGoalAreaWidth);

  const goalLPosTop = addL(halfLength - dim.goalAreaLength, halfGoalAreaWidth);
  const goalLPosBot = addL(halfLength - dim.goalAreaLength, -halfGoalAreaWidth);
  const goalLNegTop = addL(-halfLength + dim.goalAreaLength, halfGoalAreaWidth);
  const goalLNegBot = addL(-halfLength + dim.goalAreaLength, -halfGoalAreaWidth);

  edges.push(
    [goalTPosTop, goalLPosTop],
    [goalTPosBot, goalLPosBot],
    [goalLPosTop, goalLPosBot],
    [goalTNegTop, goalLNegTop],
    [goalTNegBot, goalLNegBot],
    [goalLNegTop, goalLNegBot],
  );

  const hasPenaltyArea = dim.penaltyAreaLength !== 0 && dim.penaltyAreaWidth !== 0;
  if (hasPenaltyArea) {
    const penaltyTPosTop = addT(halfLength, halfPenaltyAreaWidth);
    const penaltyTPosBot = addT(halfLength, -halfPenaltyAreaWidth);
    const penaltyTNegTop = addT(-halfLength, halfPenaltyAreaWidth);
    const penaltyTNegBot = addT(-halfLength, -halfPenaltyAreaWidth);

    const penaltyLPosTop = addL(halfLength - dim.penaltyAreaLength, halfPenaltyAreaWidth);
    const penaltyLPosBot = addL(halfLength - dim.penaltyAreaLength, -halfPenaltyAreaWidth);
    const penaltyLNegTop = addL(-halfLength + dim.penaltyAreaLength, halfPenaltyAreaWidth);
    const penaltyLNegBot = addL(-halfLength + dim.penaltyAreaLength, -halfPenaltyAreaWidth);

    edges.push(
      // Goal line sub-segments, from the pitch corner inward to the penalty and goal area T's.
      [cornerPosTop, penaltyTPosTop],
      [penaltyTPosTop, goalTPosTop],
      [goalTPosBot, penaltyTPosBot],
      [penaltyTPosBot, cornerPosBot],
      [cornerNegTop, penaltyTNegTop],
      [penaltyTNegTop, goalTNegTop],
      [goalTNegBot, penaltyTNegBot],
      [penaltyTNegBot, cornerNegBot],
      // Penalty box side and far edges.
      [penaltyTPosTop, penaltyLPosTop],
      [penaltyTPosBot, penaltyLPosBot],
      [penaltyLPosTop, penaltyLPosBot],
      [penaltyTNegTop, penaltyLNegTop],
      [penaltyTNegBot, penaltyLNegBot],
      [penaltyLNegTop, penaltyLNegBot],
    );
  } else {
    // No penalty area: the goal line runs straight from each corner to the goal area T's.
    edges.push(
      [cornerPosTop, goalTPosTop],
      [goalTPosBot, cornerPosBot],
      [cornerNegTop, goalTNegTop],
      [goalTNegBot, cornerNegBot],
    );
  }

  return { landmarks, edges };
}

// An association line's `start` should coincide almost exactly with a landmark position (both
// are computed from the same FieldDimensions) - this just guards against floating point noise.
const MATCH_TOLERANCE = 0.05;

/**
 * Matches each association line to the landmark its `start` corresponds to, returning the
 * landmark's newly detected (`end`) position keyed by landmark index. Landmarks with no
 * matching association line this frame are simply absent from the result.
 */
export function matchLandmarkEstimates(
  landmarks: FieldLandmark[],
  associationLines: { start: Vector3; end: Vector3 }[],
): Map<number, Vector3> {
  const matches = new Map<number, Vector3>();

  for (const line of associationLines) {
    let bestIndex = -1;
    let bestDistance = MATCH_TOLERANCE;
    landmarks.forEach((landmark, index) => {
      const distance = landmark.position.subtract(line.start).length;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    if (bestIndex !== -1) {
      matches.set(bestIndex, line.end);
    }
  }

  return matches;
}

export interface RigidFit2D {
  /** Rotation about Z, in radians. */
  rotation: number;
  /** Translation in the XY plane (z is always 0). */
  translation: Vector3;
}

/**
 * Computes the best-fit (least-squares) 2D rotation + translation mapping each matched
 * landmark's known ideal position onto its persisted detected position, so the *whole* known
 * field-line template can be drawn at the position/orientation implied by localisation, rather
 * than only drawing the specific points currently detected. At least 2 correspondences are
 * needed to uniquely determine a rotation; returns undefined otherwise (nothing useful to fit
 * yet).
 */
export function computeBestFitTransform(
  landmarks: FieldLandmark[],
  estimates: Map<number, Vector3>,
): RigidFit2D | undefined {
  const idealPoints: Vector3[] = [];
  const detectedPoints: Vector3[] = [];

  estimates.forEach((detected, index) => {
    idealPoints.push(landmarks[index].position);
    detectedPoints.push(detected);
  });

  if (idealPoints.length < 2) {
    return undefined;
  }

  const idealCentroid = idealPoints.reduce((sum, p) => sum.add(p), Vector3.of()).divideScalar(idealPoints.length);
  const detectedCentroid = detectedPoints
    .reduce((sum, p) => sum.add(p), Vector3.of())
    .divideScalar(detectedPoints.length);

  // Closed-form 2D Procrustes rotation (no scaling/reflection): the angle that best aligns the
  // centred ideal points onto the centred detected points.
  let sinSum = 0;
  let cosSum = 0;
  for (let i = 0; i < idealPoints.length; i++) {
    const ideal = idealPoints[i].subtract(idealCentroid);
    const detected = detectedPoints[i].subtract(detectedCentroid);
    sinSum += ideal.x * detected.y - ideal.y * detected.x;
    cosSum += ideal.x * detected.x + ideal.y * detected.y;
  }

  const rotation = Math.atan2(sinSum, cosSum);
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const rotatedCentroidX = idealCentroid.x * cos - idealCentroid.y * sin;
  const rotatedCentroidY = idealCentroid.x * sin + idealCentroid.y * cos;

  const translation = new Vector3(detectedCentroid.x - rotatedCentroidX, detectedCentroid.y - rotatedCentroidY, 0);

  return { rotation, translation };
}
