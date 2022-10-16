#include "radiance_cell.hpp"


namespace mcrt {
	RadianceCell::RadianceCell() {}

	void RadianceCell::addObject(std::shared_ptr<GameObject> obj) 
	{
		objectsInside.push_back(obj);
	}

	void RadianceCell::removeObject(std::shared_ptr<GameObject> obj)
	{
		remove(objectsInside.begin(), objectsInside.end(), obj);
	}
}