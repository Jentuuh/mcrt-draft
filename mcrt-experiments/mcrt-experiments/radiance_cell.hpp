#pragma once
#include "game_object.hpp"

#include <vector>

namespace mcrt {
	class RadianceCell
	{
	public:
		RadianceCell();

		void addObject(std::shared_ptr<GameObject> obj);
		void removeObject(std::shared_ptr<GameObject> obj);

	private:
		std::vector<std::shared_ptr<GameObject>> objectsInside;

	};
}


