#ifndef GUNDAM_MPS_SPLINE_PAYLOAD_LAYOUT_H
#define GUNDAM_MPS_SPLINE_PAYLOAD_LAYOUT_H

#include "DeviceWeightModel.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Backends {
  // Reorder whole payloads by the first logical GPU tile that uses them. Descriptor and
  // response indices remain stable; only each descriptor's payload offset moves.
  struct MpsSplinePayloadLayout {
    std::vector<DeviceSplineDialDescriptor> compact, uniform, monotonic, general, graph;
    std::vector<float> data;

    explicit MpsSplinePayloadLayout(const DeviceWeightModel& source_)
        : compact(source_.compactDialDescriptors), uniform(source_.uniformDialDescriptors),
          monotonic(source_.monotonicDialDescriptors), general(source_.generalDialDescriptors), graph(source_.graphDialDescriptors) {
      data.reserve(source_.splineData.size());
      std::array<std::vector<bool>, 5> seen{
        std::vector<bool>(compact.size()), std::vector<bool>(uniform.size()), std::vector<bool>(monotonic.size()),
        std::vector<bool>(general.size()), std::vector<bool>(graph.size())};
      const auto& ranges = source_.eventDialRanges;
      for( std::size_t first = 0; first < ranges.size(); first += 32 ){
        auto end = std::min(first + 32, ranges.size());
        visit(compact, seen[0], source_.compactDialIndices, ranges, &DeviceEventDialRanges::compactOffset, &DeviceEventDialRanges::compactCount, first, end, source_.splineData);
        visit(uniform, seen[1], source_.uniformDialIndices, ranges, &DeviceEventDialRanges::uniformOffset, &DeviceEventDialRanges::uniformCount, first, end, source_.splineData);
        visit(monotonic, seen[2], source_.monotonicDialIndices, ranges, &DeviceEventDialRanges::monotonicOffset, &DeviceEventDialRanges::monotonicCount, first, end, source_.splineData);
        visit(general, seen[3], source_.generalDialIndices, ranges, &DeviceEventDialRanges::generalOffset, &DeviceEventDialRanges::generalCount, first, end, source_.splineData);
        visit(graph, seen[4], source_.graphDialIndices, ranges, &DeviceEventDialRanges::graphOffset, &DeviceEventDialRanges::graphCount, first, end, source_.splineData);
      }
      // Include unused/dummy descriptors too; they remain legal buffer bindings.
      std::array<std::vector<DeviceSplineDialDescriptor>*, 5> families{&compact, &uniform, &monotonic, &general, &graph};
      for( std::size_t family = 0; family < families.size(); ++family ){
        for( std::size_t index = 0; index < families[family]->size(); ++index ){
          place(*families[family], seen[family], index, source_.splineData);
        }
      }
      if( data.empty() ){ data.emplace_back(0); }
    }

    void apply(DeviceWeightModel& destination_) {
      destination_.compactDialDescriptors.swap(compact);
      destination_.uniformDialDescriptors.swap(uniform);
      destination_.monotonicDialDescriptors.swap(monotonic);
      destination_.generalDialDescriptors.swap(general);
      destination_.graphDialDescriptors.swap(graph);
      destination_.splineData.swap(data);
    }

  private:
    void place(std::vector<DeviceSplineDialDescriptor>& descriptors_, std::vector<bool>& seen_,
               std::size_t index_, const std::vector<float>& original_) {
      if( seen_.at(index_) ){ return; }
      auto& descriptor = descriptors_.at(index_);
      std::size_t end = std::size_t(descriptor.splineOffset) + descriptor.splineSize;
      if( end > original_.size() or data.size() + descriptor.splineSize > std::numeric_limits<uint32_t>::max() ){
        throw std::runtime_error("Invalid or oversized MPS spline payload layout.");
      }
      auto offset = descriptor.splineOffset;
      descriptor.splineOffset = uint32_t(data.size());
      data.insert(data.end(), original_.begin() + offset, original_.begin() + end);
      seen_[index_] = true;
    }

    void visit(std::vector<DeviceSplineDialDescriptor>& descriptors_, std::vector<bool>& seen_,
               const std::vector<uint32_t>& indices_, const std::vector<DeviceEventDialRanges>& ranges_,
               DeviceIndex DeviceEventDialRanges::*offset_, DeviceIndex DeviceEventDialRanges::*count_,
               std::size_t first_, std::size_t end_, const std::vector<float>& original_) {
      uint32_t count{0};
      for( auto event = first_; event < end_; ++event ){ count = std::max(count, ranges_[event].*count_); }
      for( uint32_t dial = 0; dial < count; ++dial ){
        for( auto event = first_; event < end_; ++event ){
          const auto& range = ranges_[event];
          if( dial < range.*count_ ){ place(descriptors_, seen_, indices_.at(range.*offset_ + dial), original_); }
        }
      }
    }
  };
}
#endif
