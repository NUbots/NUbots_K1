import React from "react";
import { computed } from "mobx";
import { observer } from "mobx-react";

import { FieldDimensions } from "../../../../shared/field/dimensions";

import { RigidFit2D } from "./field/field_landmarks";
import { buildFieldLinesGeometry } from "./field/field_lines_geometry";

/**
 * Overlay showing the *whole* known field-line template (identical shape to the true white
 * field lines), positioned and rotated by `fit` - the best-fit rigid transform between each
 * known landmark's ideal position and its detected position this frame (see
 * `robot_model.ts#fieldFit`, computed in `network.ts#onField` from the robot's own best-fit
 * correspondence between detected intersections and its known field map). The whole template is
 * always drawn complete - it doesn't matter which specific corners are currently in view - but
 * its position reflects the current state of localisation: a good solution places it right on
 * top of the true field, and a bad one visibly offsets/rotates it away.
 */
@observer
export class ExpectedFieldLines extends React.Component<{
  dimensions: FieldDimensions;
  fit?: RigidFit2D;
}> {
  render() {
    const { fit } = this.props;
    if (!fit) {
      return null;
    }

    return (
      <mesh
        geometry={this.geometry}
        position={[fit.translation.x, fit.translation.y, 0.006]}
        rotation={[0, 0, fit.rotation]}
      >
        <meshBasicMaterial color="#2979ff" transparent opacity={0.85} depthWrite={false} />
      </mesh>
    );
  }

  @computed
  private get geometry() {
    return buildFieldLinesGeometry(this.props.dimensions);
  }
}
