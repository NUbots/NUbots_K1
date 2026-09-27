import React from "react";
import { computed } from "mobx";
import { observer } from "mobx-react";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils";

import { buildFieldLinesGeometry, buildRectangle } from "./field_lines_geometry";
import { FieldModel } from "./model";

@observer
export class FieldView extends React.Component<{
  model: FieldModel;
}> {
  private get model() {
    return this.props.model;
  }

  render() {
    const dim = this.model.dimensions;
    return (
      <object3D>
        <mesh receiveShadow>
          <planeGeometry
            args={[
              dim.fieldLength + dim.goalDepth * 2 + dim.borderStripMinWidth * 2,
              dim.fieldWidth + dim.borderStripMinWidth * 2,
            ]}
          />
          <meshBasicMaterial color={this.model.fieldColor} />
        </mesh>
        <mesh geometry={this.fieldLinesGeometry} position={[0, 0, 0.001]}>
          <meshBasicMaterial color={this.model.lineColor} />
        </mesh>
        <mesh geometry={this.blueHalfGoal} position={[0, 0, 0.002]}>
          <meshBasicMaterial color={this.model.blueGoalColor} />
        </mesh>
        <mesh geometry={this.yellowHalfGoal} position={[0, 0, 0.002]}>
          <meshBasicMaterial color={this.model.yellowGoalColor} />
        </mesh>
      </object3D>
    );
  }

  @computed
  private get fieldLinesGeometry() {
    return buildFieldLinesGeometry(this.model.dimensions);
  }

  @computed
  private get blueHalfGoal() {
    const lineWidth = this.model.dimensions.lineWidth;
    const goalWidth = this.model.dimensions.goalWidth;
    const goalDepth = this.model.dimensions.goalDepth;

    const halfLength = this.model.dimensions.fieldLength * 0.5;
    const halfGoalWidth = this.model.dimensions.goalWidth * 0.5;

    const blueHalfGoal = buildRectangle(
      -halfLength - goalDepth,
      -halfGoalWidth,
      goalDepth - lineWidth,
      goalWidth,
      lineWidth,
      {
        top: true,
        bottom: true,
        left: true,
        right: false,
      },
    );

    return BufferGeometryUtils.mergeGeometries([blueHalfGoal]);
  }

  @computed
  private get yellowHalfGoal() {
    const lineWidth = this.model.dimensions.lineWidth;
    const goalWidth = this.model.dimensions.goalWidth;
    const goalDepth = this.model.dimensions.goalDepth;

    const halfLength = this.model.dimensions.fieldLength * 0.5;
    const halfGoalWidth = this.model.dimensions.goalWidth * 0.5;

    const yellowHalfGoal = buildRectangle(
      halfLength + lineWidth,
      -halfGoalWidth,
      goalDepth - lineWidth,
      goalWidth,
      lineWidth,
      {
        top: true,
        bottom: true,
        left: false,
        right: true,
      },
    );

    return BufferGeometryUtils.mergeGeometries([yellowHalfGoal]);
  }
}
