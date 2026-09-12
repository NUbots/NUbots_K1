/**
 * @file FieldMapFromDescription.hpp
 * @brief Build a FieldMap/FieldDimensions from a NUbots message::support::FieldDescription.
 *
 * This is kept separate from FieldMap.hpp so the filter core (FieldMap and the measurement/system
 * headers that include it) stays free of the message library: only code that actually has a
 * FieldDescription to hand pulls in the protobuf dependency by including this header.
 */
#ifndef MODULE_LOCALISATION_SRIF_FIELDMAPFROMDESCRIPTION_HPP
#define MODULE_LOCALISATION_SRIF_FIELDMAPFROMDESCRIPTION_HPP

#include "srif/FieldMap.hpp"

#include "message/support/FieldDescription.hpp"

namespace module::localisation::srif {

    /**
     * @brief FieldDimensions from a NUbots FieldDescription message.
     *
     * Maps message::support::FieldDescription::FieldDimensions (snake_case, metres) onto the
     * estimator's FieldDimensions. Fields the map does not use (goalpost cross-section, crossbar,
     * net, etc.) are ignored.
     */
    inline FieldDimensions field_dimensions(const message::support::FieldDescription& fd) {
        FieldDimensions dims;
        dims.line_width             = fd.dimensions.line_width;
        dims.field_length           = fd.dimensions.field_length;
        dims.field_width            = fd.dimensions.field_width;
        dims.goal_depth             = fd.dimensions.goal_depth;
        dims.goal_width             = fd.dimensions.goal_width;
        dims.goal_area_length       = fd.dimensions.goal_area_length;
        dims.goal_area_width        = fd.dimensions.goal_area_width;
        dims.penalty_mark_distance  = fd.dimensions.penalty_mark_distance;
        dims.centre_circle_diameter = fd.dimensions.center_circle_diameter;
        dims.penalty_area_length    = fd.dimensions.penalty_area_length;
        dims.penalty_area_width     = fd.dimensions.penalty_area_width;
        dims.goalpost_width         = fd.dimensions.goalpost_width;
        dims.border_strip_min_width = fd.dimensions.border_strip_min_width;
        return dims;
    }

}  // namespace module::localisation::srif

#endif  // MODULE_LOCALISATION_SRIF_FIELDMAPFROMDESCRIPTION_HPP
