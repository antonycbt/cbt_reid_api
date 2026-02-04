from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class SiteHierarchy(Base):
    __tablename__ = "site_hierarchies"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(32), nullable=False)

    parent_site_hierarchy_id = Column(
        Integer,
        ForeignKey("site_hierarchies.id", ondelete="SET NULL"),
        nullable=True
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )

    # self hierarchy tree
    parent = relationship(
        "SiteHierarchy",
        remote_side=[id],
        backref="children"
    )

    # (hierarchy -> locations)
    site_locations = relationship(
        "SiteLocation",
        back_populates="site_hierarchy",
        lazy="selectin"
    )
