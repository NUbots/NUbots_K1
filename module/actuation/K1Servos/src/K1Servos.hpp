#ifndef MODULE_ACTUATION_K1SERVOS_HPP
#define MODULE_ACTUATION_K1SERVOS_HPP

#include <nuclear>

#include "extension/Behaviour.hpp"

namespace module::actuation {

    /// Single sink for the K1's low-level channel. Provides the
    /// message::actuation::K1Servos Task so the Director arbitrates ownership by
    /// priority (get-up > kick > walk) between the policy skills, then forwards the
    /// winning requester's command to the platform as a raw BoosterLowCmd. Keeps the
    /// NUgus Servos/ServoTarget path untouched — K1 has its own BoosterLowCmd path.
    class K1Servos : public ::extension::behaviour::BehaviourReactor {
    public:
        explicit K1Servos(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::actuation

#endif  // MODULE_ACTUATION_K1SERVOS_HPP
