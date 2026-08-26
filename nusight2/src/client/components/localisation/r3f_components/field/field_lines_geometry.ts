import * as THREE from "three";
import { Matrix4 } from "three";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils";

import { FieldDimensions } from "../../../../../shared/field/dimensions";

export function buildCenterCircle(dim: FieldDimensions) {
  return new THREE.RingGeometry(
    (dim.centerCircleDiameter - dim.lineWidth) * 0.5,
    (dim.centerCircleDiameter + dim.lineWidth) * 0.5,
    128,
  );
}

export function buildRectangle(
  x: number,
  y: number,
  w: number,
  h: number,
  lw: number,
  edges: { top?: boolean; bottom?: boolean; left?: boolean; right?: boolean } = {
    top: true,
    bottom: true,
    left: true,
    right: true,
  },
) {
  const x1 = x - lw * 0.5;
  const x2 = x + w + lw * 0.5;

  const geometries = [];

  if (edges.top) {
    geometries.push(buildHorizontalLine(x1, x2, y, lw));
  }
  if (edges.bottom) {
    geometries.push(buildHorizontalLine(x1, x2, y + h, lw));
  }
  if (edges.left) {
    geometries.push(buildVerticalLine(y, y + h, x, lw));
  }
  if (edges.right) {
    geometries.push(buildVerticalLine(y, y + h, x + w, lw));
  }

  return BufferGeometryUtils.mergeGeometries(geometries);
}

function buildHorizontalLine(x1: number, x2: number, y: number, width: number) {
  const length = x2 - x1;
  const hLine = new THREE.PlaneGeometry(length, width);
  hLine.applyMatrix4(new Matrix4().makeTranslation(x1 + length * 0.5, y, 0));
  return hLine;
}

function buildVerticalLine(y1: number, y2: number, x: number, width: number) {
  const length = y2 - y1;
  const vLine = new THREE.PlaneGeometry(width, length);
  vLine.applyMatrix4(new Matrix4().makeTranslation(x, y1 + length * 0.5, 0));
  return vLine;
}

/** Builds the geometry for all field markings (halfway line, center circle, penalty/goal areas, penalty marks)
 * derived purely from the known field dimensions, i.e. where the lines are expected to be. */
export function buildFieldLinesGeometry(dim: FieldDimensions) {
  const centerCircle = buildCenterCircle(dim);

  const { fieldWidth, fieldLength, lineWidth, goalAreaWidth, goalAreaLength, penaltyAreaWidth, penaltyAreaLength } =
    dim;
  const penaltyMarkDistance = dim.penaltyMarkDistance;

  const halfLength = fieldLength * 0.5;
  const halfWidth = fieldWidth * 0.5;
  const halfGoalAreaWidth = goalAreaWidth * 0.5;
  const halfPenaltyAreaWidth = penaltyAreaWidth * 0.5;

  const blueHalf = buildRectangle(-halfLength, -halfWidth, halfLength, fieldWidth, lineWidth);
  const blueHalfPenaltyArea = buildRectangle(
    -halfLength,
    -halfPenaltyAreaWidth,
    penaltyAreaLength,
    penaltyAreaWidth,
    lineWidth,
  );
  const blueHalfGoalArea = buildRectangle(-halfLength, -halfGoalAreaWidth, goalAreaLength, goalAreaWidth, lineWidth);
  const blueHalfPenaltyMark = buildRectangle(-halfLength + penaltyMarkDistance, 0, 0, 0, lineWidth);

  const yellowHalf = buildRectangle(0, -halfWidth, halfLength, fieldWidth, lineWidth);
  const yellowHalfPenaltyArea = buildRectangle(
    halfLength - penaltyAreaLength,
    -halfPenaltyAreaWidth,
    penaltyAreaLength,
    penaltyAreaWidth,
    lineWidth,
  );
  const yellowHalfGoalArea = buildRectangle(
    halfLength - goalAreaLength,
    -halfGoalAreaWidth,
    goalAreaLength,
    goalAreaWidth,
    lineWidth,
  );
  const yellowHalfPenaltyMark = buildRectangle(halfLength - penaltyMarkDistance, 0, 0, 0, lineWidth);

  return BufferGeometryUtils.mergeGeometries([
    centerCircle,
    blueHalf,
    blueHalfPenaltyArea,
    blueHalfGoalArea,
    blueHalfPenaltyMark,
    yellowHalf,
    yellowHalfPenaltyArea,
    yellowHalfGoalArea,
    yellowHalfPenaltyMark,
  ]);
}
